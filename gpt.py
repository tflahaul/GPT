import jax

from functools import partial
from typing import Tuple
from flax import linen as nn
from jax import numpy as jnp


nn.linear.default_kernel_init = nn.initializers.orthogonal(jnp.sqrt(2))


@partial(jax.jit, static_argnames=("alpha", "limit"))
def swiglu(x: jax.Array, alpha: float = 1.702, limit: float = 9.0) -> jax.Array:
	g, _x = x[..., 0::2], x[..., 1::2]
	g = jnp.clip(g, None, limit)
	_x = jnp.clip(_x, -limit, limit)
	return g * jax.nn.sigmoid(alpha * g) * (_x + 1)


@partial(jax.jit, static_argnames=("sliding_window",))
def sdpa(
	q: jax.Array,
	k: jax.Array,
	v: jax.Array,
	s: jax.Array,
	sm_scale: float,
	sliding_window: int = 0
) -> jax.Array:
	B, L, H, M, D = q.shape
	k = jnp.broadcast_to(k[:, :, :, None, :], (B, L, H, M, D))
	v = jnp.broadcast_to(v[:, :, :, None, :], (B, L, H, M, D))

	q_pos, k_pos = jnp.arange(L)[None, :, None], jnp.arange(L)[None, None, :]
	mask = nn.combine_masks(
		nn.make_causal_mask(jnp.ones(L)),
		q_pos < (k_pos + sliding_window) if sliding_window > 0 else None
	)
	attn = jnp.einsum("bqhmd,bkhmd->bhmqk", q, k) * sm_scale
	attn = jnp.where(mask, -jnp.inf, attn)
	sink = jnp.broadcast_to(s.reshape(B, H, M, 1, 1), (B, H, M, L, 1))
	w = jax.nn.softmax(jnp.concatenate([attn, sink], axis=-1))
	w = w[..., :-1]
	out = jnp.einsum("bhmqk,bkhmd->bqhmd", w, v)
	return out.reshape(B, L, -1)


class ResidualPreNorm(nn.Module):
	fn: nn.Module

	@nn.compact
	def __call__(self, x: jax.Array) -> jax.Array:
		return self.fn(nn.RMSNorm(1e-05)(x)) + x


class RotaryPositionEmbedding(nn.Module):
	""" half-truncate rotary position embedding for efficiency """
	dim: int
	max_seq_len: int
	base: int = 1024

	def setup(self) -> None:
		freqs = (1 / self.base) ** jnp.linspace(0, 1, num=(self.dim // 4))
		freqs = jnp.concatenate((freqs, jnp.zeros(self.dim // 4)))
		t = jnp.arange(self.max_seq_len, dtype=jnp.float32)
		theta = jnp.einsum("i,j->ij", t, freqs)
		self.cos_cached = jnp.expand_dims(jnp.cos(theta), (0, 2))
		self.sin_cached = jnp.expand_dims(jnp.sin(theta), (0, 2))

	def rotate(self, x: jax.Array) -> jax.Array:
		cos = self.cos_cached[:, :x.shape[1]]
		sin = self.sin_cached[:, :x.shape[1]]
		x1, x2 = jnp.split(x, 2, axis=-1)
		y1 = x1 * cos + x2 * sin
		y2 = x1 * -sin + x2 * cos
		return jnp.concatenate((y1, y2), axis=-1)

	def __call__(self, q: jax.Array, k: jax.Array) -> Tuple[jax.Array, jax.Array]:
		B, L, H, QM, D = q.shape
		q = self.rotate(jnp.reshape(q, (B, L, -1, D))).reshape((B, L, H, QM, D))
		k = self.rotate(k)
		return q, k


class GroupedQueryAttention(nn.Module):
	index: int
	n_q_heads: int
	n_kv_heads: int
	head_dim: int
	sliding_window: int
	max_seq_length: int

	@nn.compact
	def __call__(self, x: jax.Array) -> jax.Array:
		B, L, D = x.shape
		QM = self.n_q_heads // self.n_kv_heads
		s = self.param("S", nn.initializers.zeros, (1, self.n_q_heads))

		proj = nn.Dense((self.n_q_heads + 2 * self.n_kv_heads) * self.head_dim, use_bias=False)(x)
		q, k, v = jnp.split(proj, (self.n_q_heads * self.head_dim, (self.n_q_heads + self.n_kv_heads) * self.head_dim), axis=-1)
		q, k = RotaryPositionEmbedding(self.head_dim, self.max_seq_length)(
			q.reshape(B, L, self.n_kv_heads, QM, self.head_dim),
			k.reshape(B, L, self.n_kv_heads, self.head_dim)
		)
		v = v.reshape(B, L, self.n_kv_heads, self.head_dim)

		sm_scale = 1 / jnp.sqrt(self.head_dim)
		attn = sdpa(q, k, v, s, sm_scale, self.sliding_window if self.index % 2 == 0 else 0)
		out = nn.Dense(D, use_bias=False)(attn)
		return out


class MoE(nn.Module):
	n_experts: int
	n_experts_per_tok: int
	ffw_size: int
	swiglu_limit: float

	@nn.compact
	def __call__(self, x: jax.Array) -> jax.Array:
		B, L, D = x.shape
		x_f = x.reshape(B * L, D)
		gate = nn.Dense(self.n_experts, use_bias=False)(x_f)
		weights, selected_experts = jax.lax.top_k(gate, k=self.n_experts_per_tok)
		weights = jax.nn.softmax(weights, axis=-1)

		dense_1_w = self.param("dense_1_w", nn.linear.default_kernel_init, (self.n_experts, 2 * self.ffw_size, D))
		dense_1_b = self.param("dense_1_b", nn.initializers.zeros, (self.n_experts, 2 * self.ffw_size))
		dense_2_w = self.param("dense_2_w", nn.linear.default_kernel_init, (self.n_experts, D, self.ffw_size))
		dense_2_b = self.param("dense_2_b", nn.initializers.zeros, (self.n_experts, D))

		x_f = jnp.einsum("beck,bk->bec", dense_1_w[selected_experts], x_f) + dense_1_b[selected_experts]
		x_f = swiglu(x_f, limit=self.swiglu_limit)
		x_f = jnp.einsum("beck,bek->bec", dense_2_w[selected_experts], x_f) + dense_2_b[selected_experts]
		out = jnp.einsum("bec,be->bc", x_f, weights).reshape(B, L, D)
		return out


class GPT(nn.Module):
	depth: int
	n_q_heads: int
	n_kv_heads: int
	head_dim: int
	sliding_window: int
	max_seq_length: int
	n_experts: int
	n_experts_per_tok: int
	ffw_size: int
	swiglu_limit: float
	out_size: int

	@nn.compact
	def __call__(self, x: jax.Array) -> jax.Array:
		embedding = self.param("Embedding", nn.linear.default_embed_init, (self.out_size, self.ffw_size))
		x = jnp.take(embedding, x, axis=0)
		x = nn.Sequential([*(nn.Sequential([
				ResidualPreNorm(GroupedQueryAttention(i, self.n_q_heads, self.n_kv_heads, self.head_dim, self.sliding_window, self.max_seq_length)),
				ResidualPreNorm(MoE(self.n_experts, self.n_experts_per_tok, self.ffw_size, self.swiglu_limit))
			]) for i in range(self.depth))])(x)
		x = nn.RMSNorm(1e-05)(x)
		x = x @ embedding.T  # inverse embedding
		return x
