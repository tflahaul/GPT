import jax

from functools import partial
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
	k = jnp.broadcast_to(k[:, :, :, None], (B, L, H, M, D))
	v = jnp.broadcast_to(v[:, :, :, None], (B, L, H, M, D))

	q_pos, k_pos = jnp.arange(L)[None, :, None], jnp.arange(L)[None, None, :]
	mask = nn.combine_masks(
		nn.make_causal_mask(jnp.ones(L)),
		q_pos < (k_pos + sliding_window) if sliding_window > 0 else None
	)
	attn = jnp.einsum("bqhmd,bkhmd->bhmqk", q, k) * sm_scale
	attn = jnp.where(mask, -jnp.inf, attn)
	sink = jnp.broadcast_to(s[..., None, None], attn.shape[:-1] + (1,))
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
	dim: int
	max_seq_len: int

	def setup(self) -> None:
		inv_freq = (1 / 1024) ** jnp.linspace(0, 1, num=(self.dim // 4))
		inv_freq = jnp.concatenate((inv_freq, jnp.zeros(self.dim // 4)))
		t = jnp.arange(self.max_seq_len, dtype=jnp.float32)
		theta = jnp.einsum("i,j->ij", t, inv_freq)
		self.cos, self.sin = jnp.cos(theta), jnp.sin(theta)

	def __call__(self, x: jax.Array) -> jax.Array:
		cos = self.cos[None, :x.shape[1], None, :]
		sin = self.sin[None, :x.shape[1], None, :]
		x1, x2 = jnp.split(x, 2, axis=-1)
		return jnp.concatenate([
			x1 * cos + x2 * sin,
			x1 * -sin + x2 * cos
		], axis=-1)


class GroupedQueryAttention(nn.Module):
	index: int
	n_q_heads: int
	n_kv_heads: int
	head_dim: int
	sliding_window: int

	@nn.compact
	def __call__(self, x: jax.Array) -> jax.Array:
		B, L, D = x.shape
		QM = self.n_q_heads // self.n_kv_heads
		s = self.param("S", nn.initializers.zeros, (1, self.n_kv_heads, QM))

		proj = nn.Dense((self.n_q_heads + 2 * self.n_kv_heads) * self.head_dim, use_bias=False)(x)
		q, k, v = jnp.split(proj, (self.n_q_heads * self.head_dim, (self.n_q_heads + self.n_kv_heads) * self.head_dim), axis=-1)
		q = RotaryPositionEmbedding(self.head_dim, L)(q.reshape(B, L, self.n_kv_heads, QM, self.head_dim))
		k = RotaryPositionEmbedding(self.head_dim, L)(k.reshape(B, L, self.n_kv_heads, self.head_dim))
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
	n_experts: int
	n_experts_per_tok: int
	ffw_size: int
	swiglu_limit: float
	out_size: int

	@nn.compact
	def __call__(self, x: jax.Array) -> jax.Array:
		x = nn.Sequential([*(nn.Sequential([
				ResidualPreNorm(GroupedQueryAttention(i, self.n_q_heads, self.n_kv_heads, self.head_dim, self.sliding_window)),
				ResidualPreNorm(MoE(self.n_experts, self.n_experts_per_tok, self.ffw_size, self.swiglu_limit))
			]) for i in range(self.depth))])(x)
		x = nn.RMSNorm(1e-05)(x)
		x = nn.Dense(self.out_size, use_bias=False)(x)
		return x
