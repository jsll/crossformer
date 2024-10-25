# adapted from https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py
from typing import Callable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from crossformer.model.components.base import TokenGroup
from crossformer.utils.typing import Dtype, PRNGKey, Shape, Union

import functools
import warnings
from typing import Any, Callable, Optional, Tuple, Union, overload

import jax
import jax.numpy as jnp
from jax import lax, random

from flax.linen import initializers
from flax.linen.dtypes import promote_dtype
from flax.linen.linear import (
  DenseGeneral,
  DotGeneralT,
  PrecisionLike,
  default_kernel_init,
)
from flax.linen.module import Module, compact, merge_param
from flax.linen.normalization import LayerNorm

PRNGKey = jax.Array
Shape = Tuple[int, ...]
Dtype = Any
Array = Any

def dot_product_attention_weights(
  query: Array,
  key: Array,
  bias: Optional[Array] = None,
  mask: Optional[Array] = None,
  broadcast_dropout: bool = True,
  dropout_rng: Optional[PRNGKey] = None,
  dropout_rate: float = 0.0,
  deterministic: bool = False,
  dtype: Optional[Dtype] = None,
  precision: PrecisionLike = None,
  module: Optional[Module] = None,
):
  """Computes dot-product attention weights given query and key.

  Used by :func:`dot_product_attention`, which is what you'll most likely use.
  But if you want access to the attention weights for introspection, then
  you can directly call this function and call einsum yourself.

  Args:
    query: queries for calculating attention with shape of `[batch..., q_length,
      num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of `[batch..., kv_length,
      num_heads, qk_depth_per_head]`.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`. This can be used for
      incorporating causal masks, padding masks, proximity bias, etc.
    mask: mask for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`. This can be used for
      incorporating causal masks. Attention weights are masked out if their
      corresponding mask value is `False`.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: infer from inputs and params)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    module: the Module that will sow the attention weights into the
      'intermediates' collection. Remember to mark 'intermediates' as mutable via
      `mutable=['intermediates'] in order to have that collection returned.
      If `module` is None, the attention weights will not be sowed.

  Returns:
    Output of shape `[batch..., num_heads, q_length, kv_length]`.
  """
  query, key = promote_dtype(query, key, dtype=dtype)
  dtype = query.dtype

  assert query.ndim == key.ndim, 'q, k must have same rank.'
  assert query.shape[:-3] == key.shape[:-3], 'q, k batch dims must match.'
  assert query.shape[-2] == key.shape[-2], 'q, k num_heads must match.'
  assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

  # calculate attention matrix
  depth = query.shape[-1]
  query = query / jnp.sqrt(depth).astype(dtype)
  # attn weight shape is (batch..., num_heads, q_length, kv_length)
  attn_weights = jnp.einsum(
    '...qhd,...khd->...hqk', query, key, precision=precision
  )

  # apply attention bias: masking, dropout, proximity bias, etc.
  if bias is not None:
    attn_weights = attn_weights + bias
  # apply attention mask
  if mask is not None:
    big_neg = jnp.finfo(dtype).min
    attn_weights = jnp.where(mask, attn_weights, big_neg)

  # normalize the attention weights
  attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

  if module:
    module.sow('intermediates', 'attention_weights', attn_weights)

  # apply attention dropout
  if not deterministic and dropout_rate > 0.0:
    keep_prob = 1.0 - dropout_rate
    if broadcast_dropout:
      # dropout is broadcast across the batch + head dimensions
      dropout_shape = tuple([1] * (key.ndim - 2)) + attn_weights.shape[-2:]
      keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)  # type: ignore
    else:
      keep = random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)  # type: ignore
    multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
    attn_weights = attn_weights * multiplier

  return attn_weights



def dot_product_attention(
  query: Array,
  key: Array,
  value: Array,
  bias: Optional[Array] = None,
  mask: Optional[Array] = None,
  broadcast_dropout: bool = True,
  dropout_rng: Optional[PRNGKey] = None,
  dropout_rate: float = 0.0,
  deterministic: bool = False,
  dtype: Optional[Dtype] = None,
  precision: PrecisionLike = None,
  module: Optional[Module] = None,
):
  """Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights.

  Note: query, key, value needn't have any batch dimensions.

  Args:
    query: queries for calculating attention with shape of `[batch..., q_length,
      num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of `[batch..., kv_length,
      num_heads, qk_depth_per_head]`.
    value: values to be used in attention with shape of `[batch..., kv_length,
      num_heads, v_depth_per_head]`.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`. This can be used for
      incorporating causal masks, padding masks, proximity bias, etc.
    mask: mask for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`. This can be used for
      incorporating causal masks. Attention weights are masked out if their
      corresponding mask value is `False`.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: infer from inputs)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    module: the Module that will sow the attention weights into the
      'intermediates' collection. Remember to mark 'intermediates' as mutable via
      `mutable=['intermediates'] in order to have that collection returned.
      If `module` is None, the attention weights will not be sowed.

  Returns:
    Output of shape `[batch..., q_length, num_heads, v_depth_per_head]`.
  """
  query, key, value = promote_dtype(query, key, value, dtype=dtype)
  dtype = query.dtype
  assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
  assert (
    query.shape[:-3] == key.shape[:-3] == value.shape[:-3]
  ), 'q, k, v batch dims must match.'
  assert (
    query.shape[-2] == key.shape[-2] == value.shape[-2]
  ), 'q, k, v num_heads must match.'
  assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'

  # compute attention weights
  attn_weights = dot_product_attention_weights(
    query,
    key,
    bias,
    mask,
    broadcast_dropout,
    dropout_rng,
    dropout_rate,
    deterministic,
    dtype,
    precision,
    module,
  )

  # return weighted sum over values for each query position
  return jnp.einsum(
    '...hqk,...khd->...qhd', attn_weights, value, precision=precision
  )


class MultiHeadDotProductAttention(Module):
  """Multi-head dot-product attention.

  Example usage::

    >>> import flax.linen as nn
    >>> import jax

    >>> layer = nn.MultiHeadDotProductAttention(num_heads=8, qkv_features=16)
    >>> key1, key2, key3, key4, key5, key6 = jax.random.split(jax.random.key(0), 6)
    >>> shape = (4, 3, 2, 5)
    >>> q, k, v = jax.random.uniform(key1, shape), jax.random.uniform(key2, shape), jax.random.uniform(key3, shape)
    >>> variables = layer.init(jax.random.key(0), q)

    >>> # different inputs for inputs_q, inputs_k and inputs_v
    >>> out = layer.apply(variables, q, k, v)
    >>> # equivalent to layer.apply(variables, inputs_q=q, inputs_k=k, inputs_v=k)
    >>> out = layer.apply(variables, q, k)
    >>> # equivalent to layer.apply(variables, inputs_q=q, inputs_k=q) and layer.apply(variables, inputs_q=q, inputs_k=q, inputs_v=q)
    >>> out = layer.apply(variables, q)

    >>> attention_kwargs = dict(
    ...     num_heads=8,
    ...     qkv_features=16,
    ...     kernel_init=nn.initializers.ones,
    ...     bias_init=nn.initializers.zeros,
    ...     dropout_rate=0.5,
    ...     deterministic=False,
    ...     )
    >>> class Module(nn.Module):
    ...   attention_kwargs: dict
    ...
    ...   @nn.compact
    ...   def __call__(self, x, dropout_rng=None):
    ...     out1 = nn.MultiHeadDotProductAttention(**self.attention_kwargs)(x, dropout_rng=dropout_rng)
    ...     out2 = nn.MultiHeadDotProductAttention(**self.attention_kwargs)(x, dropout_rng=dropout_rng)
    ...     return out1, out2
    >>> module = Module(attention_kwargs)
    >>> variables = module.init({'params': key1, 'dropout': key2}, q)

    >>> # out1 and out2 are different.
    >>> out1, out2 = module.apply(variables, q, rngs={'dropout': key3})
    >>> # out3 and out4 are different.
    >>> # out1 and out3 are different. out2 and out4 are different.
    >>> out3, out4 = module.apply(variables, q, rngs={'dropout': key4})
    >>> # out1 and out2 are the same.
    >>> out1, out2 = module.apply(variables, q, dropout_rng=key5)
    >>> # out1 and out2 are the same as out3 and out4.
    >>> # providing a `dropout_rng` arg will take precedence over the `rngs` arg in `.apply`
    >>> out3, out4 = module.apply(variables, q, rngs={'dropout': key6}, dropout_rng=key5)

  Attributes:
    num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
      should be divisible by the number of heads.
    dtype: the dtype of the computation (default: infer from inputs and params)
    param_dtype: the dtype passed to parameter initializers (default: float32)
    qkv_features: dimension of the key, query, and value.
    out_features: dimension of the last projection
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rate: dropout rate
    deterministic: if false, the attention weight is masked randomly using
      dropout, whereas if true, the attention weights are deterministic.
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer for the kernel of the Dense layers.
    bias_init: initializer for the bias of the Dense layers.
    use_bias: bool: whether pointwise QKVO dense transforms use bias.
    attention_fn: dot_product_attention or compatible function. Accepts query,
      key, value, and returns output of shape `[bs, dim1, dim2, ..., dimN,,
      num_heads, value_channels]``
    decode: whether to prepare and use an autoregressive cache.
    normalize_qk: should QK normalization be applied (arxiv.org/abs/2302.05442).
  """

  num_heads: int
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  qkv_features: Optional[int] = None
  out_features: Optional[int] = None
  broadcast_dropout: bool = True
  dropout_rate: float = 0.0
  deterministic: Optional[bool] = None
  precision: PrecisionLike = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[
    [PRNGKey, Shape, Dtype], Array
  ] = initializers.zeros_init()
  use_bias: bool = True
  attention_fn: Callable[..., Array] = dot_product_attention
  decode: bool = False
  normalize_qk: bool = False
  # Deprecated, will be removed.
  qkv_dot_general: Optional[DotGeneralT] = None
  out_dot_general: Optional[DotGeneralT] = None
  qkv_dot_general_cls: Any = None
  out_dot_general_cls: Any = None

  @overload
  def __call__(
    self,
    inputs_q: Array,
    inputs_k: Optional[Array] = None,
    inputs_v: Optional[Array] = None,
    *,
    mask: Optional[Array] = None,
    deterministic: Optional[bool] = None,
    dropout_rng: Optional[PRNGKey] = None,
    return_weights: bool = False,
  ):
    ...

  @overload
  def __call__(
    self,
    inputs_q: Array,
    *,
    inputs_kv: Array = None,
    mask: Optional[Array] = None,
    deterministic: Optional[bool] = None,
    dropout_rng: Optional[PRNGKey] = None,
    return_weights: bool = False,
  ):
    ...

  @compact
  def __call__(
    self,
    inputs_q: Array,
    inputs_k: Optional[Array] = None,
    inputs_v: Optional[Array] = None,
    *,
    inputs_kv: Optional[Array] = None,
    mask: Optional[Array] = None,
    deterministic: Optional[bool] = None,
    dropout_rng: Optional[PRNGKey] = None,
    return_weights: bool = False,
  ):
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    If both inputs_k and inputs_v are None, they will both copy the value of
    inputs_q (self attention).
    If only inputs_v is None, it will copy the value of inputs_k.

    Args:
      inputs_q: input queries of shape `[batch_sizes..., length, features]`.
      inputs_k: key of shape `[batch_sizes..., length, features]`. If None,
        inputs_k will copy the value of inputs_q.
      inputs_v: values of shape `[batch_sizes..., length, features]`. If None,
        inputs_v will copy the value of inputs_k.
      inputs_kv: key/values of shape `[batch_sizes..., length, features]`. If
        None, inputs_kv will copy the value of inputs_q. This arg will be
        deprecated soon. Use inputs_k and inputs_v instead.
      mask: attention mask of shape `[batch_sizes..., num_heads, query_length,
        key/value_length]`. Attention weights are masked out if their
        corresponding mask value is `False`.
      deterministic: if false, the attention weight is masked randomly using
        dropout, whereas if true, the attention weights are deterministic.
      dropout_rng: optional rng key to pass to the attention layer's dropout
        mask. Otherwise, self.make_rng('dropout') is used instead.
      return_weights: if `True`, the attention weights are sowed into the
        'intermediates' collection. Remember to mark 'intermediates' as
        mutable via `mutable=['intermediates'] in order to have that
        collection returned.

    Returns:
      output of shape `[batch_sizes..., length, features]`.
    """
    if inputs_kv is not None:
      if inputs_k is not None or inputs_v is not None:
        raise ValueError(
          'If either `inputs_k` or `inputs_v` is not None, '
          '`inputs_kv` must be None. If `inputs_kv` is not None, both `inputs_k` '
          'and `inputs_v` must be None. We recommend using `inputs_k` and '
          '`inputs_v` args, since `inputs_kv` will be deprecated soon. See '
          'https://github.com/google/flax/discussions/3389 for more '
          'information.'
        )
      inputs_k = inputs_v = inputs_kv
      warnings.warn(
        'The inputs_kv arg will be deprecated soon. '
        'Use inputs_k and inputs_v instead. See '
        'https://github.com/google/flax/discussions/3389 '
        'for more information.',
        DeprecationWarning,
      )
    else:
      if inputs_k is None:
        if inputs_v is not None:
          raise ValueError(
            '`inputs_k` cannot be None if `inputs_v` is not None. '
            'To have both `inputs_k` and `inputs_v` be the same value, pass in the '
            'value to `inputs_k` and leave `inputs_v` as None.'
          )
        inputs_k = inputs_q
      if inputs_v is None:
        inputs_v = inputs_k
      elif inputs_v.shape[-1] == inputs_v.shape[-2]:
        warnings.warn(
          f'You are passing an array of shape {inputs_v.shape} '
          'to the `inputs_v` arg, when you may have intended '
          'to pass it to the `mask` arg. As of Flax version '
          '0.7.4, the function signature of '
          "MultiHeadDotProductAttention's `__call__` method "
          'has changed to `__call__(inputs_q, inputs_k=None, '
          'inputs_v=None, *, inputs_kv=None, mask=None, '
          'deterministic=None)`. Use the kwarg `mask` instead. '
          'See https://github.com/google/flax/discussions/3389 '
          'and read the docstring for more information.',
          DeprecationWarning,
        )

    features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    assert qkv_features % self.num_heads == 0, (
      f'Memory dimension ({qkv_features}) must be divisible by number of'
      f' heads ({self.num_heads}).'
    )
    head_dim = qkv_features // self.num_heads

    dense = functools.partial(
      DenseGeneral,
      axis=-1,
      dtype=self.dtype,
      param_dtype=self.param_dtype,
      features=(self.num_heads, head_dim),
      kernel_init=self.kernel_init,
      bias_init=self.bias_init,
      use_bias=self.use_bias,
      precision=self.precision,
      dot_general=self.qkv_dot_general,
      dot_general_cls=self.qkv_dot_general_cls,
    )
    # project inputs_q to multi-headed q/k/v
    # dimensions are then [batch..., length, n_heads, n_features_per_head]
    query, key, value = (
      dense(name='query')(inputs_q),
      dense(name='key')(inputs_k),
      dense(name='value')(inputs_v),
    )

    if self.normalize_qk:
      # Normalizing query and key projections stabilizes training with higher
      # LR. See ViT-22B paper http://arxiv.org/abs/2302.05442 for analysis.
      query = LayerNorm(name='query_ln', use_bias=False)(query)  # type: ignore[call-arg]
      key = LayerNorm(name='key_ln', use_bias=False)(key)  # type: ignore[call-arg]

    # During fast autoregressive decoding, we feed one position at a time,
    # and cache the keys and values step by step.
    if self.decode:
      # detect if we're initializing by absence of existing cache data.
      is_initialized = self.has_variable('cache', 'cached_key')
      cached_key = self.variable(
        'cache', 'cached_key', jnp.zeros, key.shape, key.dtype
      )
      cached_value = self.variable(
        'cache', 'cached_value', jnp.zeros, value.shape, value.dtype
      )
      cache_index = self.variable(
        'cache', 'cache_index', lambda: jnp.array(0, dtype=jnp.int32)
      )
      if is_initialized:
        (
          *batch_dims,
          max_length,
          num_heads,
          depth_per_head,
        ) = cached_key.value.shape
        # shape check of cached keys against query input
        expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
        if expected_shape != query.shape:
          raise ValueError(
            'Autoregressive cache shape error, '
            'expected query shape %s instead got %s.'
            % (expected_shape, query.shape)
          )
        # update key, value caches with our new 1d spatial slices
        cur_index = cache_index.value
        indices: tuple[Union[int, jax.Array], ...] = (0,) * len(batch_dims) + (
          cur_index,
          0,
          0,
        )
        key = lax.dynamic_update_slice(cached_key.value, key, indices)
        value = lax.dynamic_update_slice(cached_value.value, value, indices)
        cached_key.value = key
        cached_value.value = value
        cache_index.value = cache_index.value + 1
        # causal mask for cached decoder self-attention:
        # our single query position should only attend to those key
        # positions that have already been generated and cached,
        # not the remaining zero elements.
        mask = combine_masks(
          mask,
          jnp.broadcast_to(
            jnp.arange(max_length) <= cur_index,
            tuple(batch_dims) + (1, 1, max_length),
          ),
        )

    if (
      self.dropout_rate > 0.0
    ):  # Require `deterministic` only if using dropout.
      m_deterministic = merge_param(
        'deterministic', self.deterministic, deterministic
      )
      if not m_deterministic and dropout_rng is None:
        dropout_rng = self.make_rng('dropout')
    else:
      m_deterministic = True

    # apply attention
    if return_weights:
      x = self.attention_fn(
        query,
        key,
        value,
        mask=mask,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        broadcast_dropout=self.broadcast_dropout,
        deterministic=m_deterministic,
        dtype=self.dtype,
        precision=self.precision,
        module=self if return_weights else None,
      )  # pytype: disable=wrong-keyword-args
    else:
      x = self.attention_fn(
        query,
        key,
        value,
        mask=mask,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        broadcast_dropout=self.broadcast_dropout,
        deterministic=m_deterministic,
        dtype=self.dtype,
        precision=self.precision,
      )
    # back to the original inputs dimensions
    out = DenseGeneral(
      features=features,
      axis=(-2, -1),
      kernel_init=self.kernel_init,
      bias_init=self.bias_init,
      use_bias=self.use_bias,
      dtype=self.dtype,
      param_dtype=self.param_dtype,
      precision=self.precision,
      dot_general=self.out_dot_general,
      dot_general_cls=self.out_dot_general_cls,
      name='out',  # type: ignore[call-arg]
    )(x)
    return out




class AddPositionEmbs(nn.Module):
    """Adds learned positional embeddings to the inputs.

    Attributes:
      posemb_init: positional embedding initializer.
    """

    posemb_init: Callable[[PRNGKey, Shape, Dtype], jax.Array]

    @nn.compact
    def __call__(self, inputs):
        """Applies the AddPositionEmbs module.

        Args:
          inputs: Inputs to the layer.

        Returns:
          Output tensor with shape `(bs, timesteps, in_dim)`.
        """
        # inputs.shape is (batch_size, seq_len, emb_dim).
        assert inputs.ndim == 3, (
            "Number of dimensions should be 3," " but it is: %d" % inputs.ndim
        )
        pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
        pe = self.param("pos_embedding", self.posemb_init, pos_emb_shape)
        return inputs + pe


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int
    dtype: Dtype = jnp.float32
    out_dim: Optional[int] = None
    dropout_rate: float = 0.1
    kernel_init: Callable[
        [PRNGKey, Shape, Dtype], jax.Array
    ] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape, Dtype], jax.Array] = nn.initializers.normal(
        stddev=1e-6
    )

    @nn.compact
    def __call__(self, inputs, *, deterministic):
        """Applies Transformer MlpBlock module."""
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(
            features=self.mlp_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(inputs)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        output = nn.Dense(
            features=actual_out_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)
        output = nn.Dropout(rate=self.dropout_rate)(output, deterministic=deterministic)
        return output


class MAPHead(nn.Module):
    """Multihead Attention Pooling.

    From https://github.com/google-research/big_vision/blob/main/big_vision/models/vit.py
    """

    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    num_heads: int = 8
    num_readouts: int = 1

    @nn.compact
    def __call__(self, x: Union[jax.Array, TokenGroup], train=True):
        if isinstance(x, TokenGroup):
            x, mask = x.tokens, x.mask
        else:
            mask = None

        *batch_dims, l, d = x.shape
        x = x.reshape(-1, l, d)
        batch_size = x.shape[0]

        probe = self.param(
            "probe",
            nn.initializers.xavier_uniform(),
            (1, self.num_readouts, d),
            x.dtype,
        )
        probe = jnp.tile(probe, [batch_size, 1, 1])

        if mask is not None:
            mask = mask.reshape(-1, l)
            mask = jnp.broadcast_to(
                mask[:, None, None, :], (batch_size, 1, self.num_readouts, l)
            )

        out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, kernel_init=nn.initializers.xavier_uniform()
        )(probe, x, mask=mask)

        # TODO: dropout on head?
        y = nn.LayerNorm()(out)

        out = out + MlpBlock(mlp_dim=nn.merge_param("mlp_dim", self.mlp_dim, 4 * d))(
            y, deterministic=not train
        )
        out = out.reshape(*batch_dims, self.num_readouts, d)
        return out


class Encoder1DBlock(nn.Module):
    """Transformer encoder layer.

    Attributes:
      inputs: input data.
      mlp_dim: dimension of the mlp on top of attention block.
      dtype: the dtype of the computation (default: float32).
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout for attention heads.
      deterministic: deterministic or not (to apply dropout).
      num_heads: Number of heads in nn.MultiHeadDotProductAttention
      repeat_pos_enc: should pos be added back to keys/values during attention?
    """

    mlp_dim: int
    num_heads: int
    dtype: Dtype = jnp.float32
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    repeat_pos_enc: bool = False

    @nn.compact
    def __call__(self, inputs, pos_enc, attention_mask, *, deterministic):
        """Applies Encoder1DBlock module.

        Args:
          inputs: Inputs to the layer.
          deterministic: Dropout will not be applied when set to true.

        Returns:
          output after transformer encoder block.
        """

        # Attention block.
        assert inputs.ndim == 3, f"Expected (batch, seq, hidden) got {inputs.shape}"
        x = nn.LayerNorm(dtype=self.dtype)(inputs)

        qk = x + pos_enc if self.repeat_pos_enc else x
        x = MultiHeadDotProductAttention(
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            broadcast_dropout=False,
            deterministic=deterministic,
            dropout_rate=self.attention_dropout_rate,
            num_heads=self.num_heads,
        )(inputs_q=qk, inputs_k=qk, inputs_v=x, mask=attention_mask, return_weights=True)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = x + inputs

        # MLP block.
        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = MlpBlock(
            mlp_dim=self.mlp_dim, dtype=self.dtype, dropout_rate=self.dropout_rate
        )(y, deterministic=deterministic)

        return x + y


class Transformer(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation.

    Attributes:
      num_layers: number of layers
      mlp_dim: dimension of the mlp on top of attention block
      num_heads: Number of heads in nn.MultiHeadDotProductAttention
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout rate in self attention.
      repeat_pos_enc: positional encodings added back to k/q during attn.
    """

    num_layers: int
    mlp_dim: int
    num_attention_heads: int
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    repeat_pos_enc: bool = False

    @nn.compact
    def __call__(self, x, pos_enc, attention_mask, *, train):
        """Applies Transformer model on the inputs.

        Args:
          x: Inputs to the layer.
          train: Set to `True` when training.

        Returns:
          output of a transformer encoder.
        """
        assert x.ndim == 3  # (batch, len, emb)

        # only add pos_enc at start of computation if not repeating
        if not self.repeat_pos_enc:
            x = x + pos_enc

        # Input Encoder
        for lyr in range(self.num_layers):
            x = Encoder1DBlock(
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                name=f"encoderblock_{lyr}",
                num_heads=self.num_attention_heads,
                repeat_pos_enc=self.repeat_pos_enc,
            )(x, pos_enc, attention_mask, deterministic=not train)
        encoded = nn.LayerNorm(name="encoder_norm")(x)

        return encoded


def common_transformer_sizes(transformer_size: str) -> (int, dict):
    """
    Args:
        transformer_size (str): The size of the transformer. One of "dummy", "vanilla", "detr", "vit_s", "vit_b", "vit_l", "vit_h"

    Returns:
            token_embedding_size (int): The size of the token embeddings
            transformer_kwargs (dict): The kwargs to pass to the transformer

    """
    assert transformer_size in [
        "dummy",
        "vanilla",
        "detr",
        "vit_t",
        "vit_s",
        "vit_b",
        "vit_l",
        "vit_h",
        "vint",
        "vit_t_repeat",
        "vit_s_repeat",
        "detr_big",
    ]
    default_params = {
        "attention_dropout_rate": 0.0,
    }

    TRANSFORMER_SIZES = {
        "dummy": dict(
            num_layers=1,
            mlp_dim=256,
            num_attention_heads=2,
            dropout_rate=0.1,
            repeat_pos_enc=False,
        ),
        "vanilla": dict(
            num_layers=4,
            mlp_dim=1024,
            num_attention_heads=8,
            dropout_rate=0.1,
            repeat_pos_enc=False,
        ),
        "detr": dict(
            num_layers=12,  # techincally detr uses 6 enc + 6 dec
            mlp_dim=2048,
            num_attention_heads=8,
            dropout_rate=0.1,
            repeat_pos_enc=True,
        ),
        "detr_big": dict(
            num_layers=24,  # techincally detr uses 6 enc + 6 dec
            mlp_dim=4096,
            num_attention_heads=16,
            dropout_rate=0.1,
            repeat_pos_enc=True,
        ),
        "vit_t": dict(
            num_layers=12,
            mlp_dim=768,
            num_attention_heads=3,
            dropout_rate=0.0,
            repeat_pos_enc=False,
        ),
        "vit_s": dict(
            num_layers=12,
            mlp_dim=1536,
            num_attention_heads=6,
            dropout_rate=0.0,
            repeat_pos_enc=False,
        ),
        "vit_b": dict(
            num_layers=12,
            mlp_dim=3072,
            num_attention_heads=12,
            dropout_rate=0.0,
            repeat_pos_enc=False,
        ),
        "vit_l": dict(
            num_layers=24,
            mlp_dim=4096,
            num_attention_heads=16,
            dropout_rate=0.1,
            repeat_pos_enc=False,
        ),
        "vit_h": dict(
            num_layers=32,
            mlp_dim=5120,
            num_attention_heads=16,
            dropout_rate=0.1,
            repeat_pos_enc=False,
        ),
        "vint": dict(
            num_layers=4,
            mlp_dim=2048,
            num_attention_heads=4,
            dropout_rate=0.0,
        ),
        "vit_t_repeat": dict(
            num_layers=12,
            mlp_dim=768,
            num_attention_heads=3,
            dropout_rate=0.1,
            repeat_pos_enc=True,
        ),
        "vit_s_repeat": dict(
            num_layers=12,
            mlp_dim=1536,
            num_attention_heads=6,
            dropout_rate=0.1,
            repeat_pos_enc=True,
        ),
    }

    TOKEN_DIMS = {
        "dummy": 256,
        "vanilla": 256,
        "detr": 512,
        "detr_big": 1024,
        "vit_t": 192,
        "vit_s": 384,
        "vit_b": 768,
        "vit_l": 1024,
        "vit_h": 1280,
        "vint": 512,
        "vit_t_repeat": 192,
        "vit_s_repeat": 384,
    }

    return TOKEN_DIMS[transformer_size], {
        **default_params,
        **TRANSFORMER_SIZES[transformer_size],
    }


