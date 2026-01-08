import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

class CausalQueue(keras.layers.Layer):
    def __init__(self, dilation_rate):
        super(CausalQueue, self).__init__()
        self._dilation_rate = dilation_rate
        self.stateful = True
        
    def build(self, input_shape):
        # We need a fixed batch size to hold state.
        if input_shape[0] is None:
            raise ValueError("CausalQueue requires a fixed batch size (batch dimension must not be None).")
        if input_shape[1] != 1 and input_shape[1] is not None:
            raise ValueError(f"CausalQueue expects time dimension = 1 (got {input_shape[1]}).")
        if input_shape[2] is None:
            raise ValueError("CausalQueue requires a known channel dimension (last dim must not be None).")

        # Keep the last `dilation_rate` frames; return [t-d, t] for kernel=2 conv.
        #
        # IMPORTANT: This is *state*, not a model weight. We intentionally do NOT track it
        # as a Keras weight so that old checkpoints (saved before buffers existed) can load
        # without mismatch errors.
        # Variable creation must be lifted out of tf.function tracing context.
        # Keras may build layers while tracing graphs (e.g., with KerasTensor), so we use init_scope.
        shape = (int(input_shape[0]), int(self._dilation_rate), int(input_shape[2]))
        with tf.init_scope():
            buf = tf.Variable(
                initial_value=lambda: tf.zeros(shape, dtype=self.compute_dtype),
                trainable=False,
                name="buffer",
            )
        self._buffer = self._no_dependency(buf)
        super(CausalQueue, self).build(input_shape)
        
    def reset_states(self):
        self._buffer.assign(tf.zeros_like(self._buffer))
        
    def call(self, x):
        x = tf.cast(x, self.compute_dtype)
        # Stateful mode is intended for step-by-step generation.
        tf.debugging.assert_equal(
            tf.shape(x)[1], 1,
            message="CausalQueue expects time dimension = 1 per call in stateful mode."
        )

        # Oldest element is t-d.
        x_past = self._buffer[:, :1, :]

        # Shift left and append current x.
        self._buffer.assign(tf.concat((self._buffer[:, 1:, :], x), axis=1))

        # Return [t-d, t] (time dimension = 2)
        return tf.concat((x_past, x), axis=1)
    
class ShiftBuffer(keras.layers.Layer):
    def __init__(self, width):
        super(ShiftBuffer, self).__init__()
        self.width = width
        
    def build(self, input_shape):
        if input_shape[0] is None:
            raise ValueError("ShiftBuffer requires a fixed batch size (batch dimension must not be None).")
        if input_shape[1] != 1 and input_shape[1] is not None:
            raise ValueError(f"ShiftBuffer expects time dimension = 1 (got {input_shape[1]}).")
        if input_shape[2] is None:
            raise ValueError("ShiftBuffer requires a known channel dimension (last dim must not be None).")

        # IMPORTANT: This is *state*, not a model weight. We intentionally do NOT track it
        # as a Keras weight so that old checkpoints can load without mismatch errors.
        # Variable creation must be lifted out of tf.function tracing context.
        shape = (int(input_shape[0]), int(self.width), int(input_shape[2]))
        with tf.init_scope():
            buf = tf.Variable(
                initial_value=lambda: tf.zeros(shape, dtype=self.compute_dtype),
                trainable=False,
                name="buffer",
            )
        self.buffer = self._no_dependency(buf)
        super(ShiftBuffer, self).build(input_shape)
        
    def reset_states(self):
        self.buffer.assign(tf.zeros_like(self.buffer))
        
    def call(self, x):
        x = tf.cast(x, self.compute_dtype)
        tf.debugging.assert_equal(
            tf.shape(x)[1], 1,
            message="ShiftBuffer expects time dimension = 1 per call in stateful mode."
        )
        self.buffer.assign(tf.concat((self.buffer[:, 1:], x), 1))
        return self.buffer


class CausalConv1D(keras.layers.Layer):
    """
    Causal 1D convolution that supports both training and stateful inference modes.
    
    In training mode (stateful=False): Uses padding='causal' for efficient parallel processing.
    In inference mode (stateful=True): Uses CausalQueue (kernel=2) or ShiftBuffer (kernel>2)
                                        for step-by-step autoregressive generation.
    """
    def __init__(self, channels, kernel, dilation=1, activation=None, stateful=False, 
                 bias_initializer='zeros', name=None, **kwargs):
        super(CausalConv1D, self).__init__(name=name, **kwargs)
        self.channels = channels
        self.kernel = kernel
        self.dilation = dilation
        self.activation = activation
        self.stateful = stateful
        self.bias_initializer = bias_initializer
        
        # Effective receptive field size
        self.receptive_field = (kernel - 1) * dilation + 1
        
        # These will be created in build()
        self._buffer = None
        self._conv = None
    
    def build(self, input_shape):
        if self.stateful:
            # For kernel=2: CausalQueue handles dilation, conv uses dilation=1
            # For kernel>2: ShiftBuffer holds full receptive field, conv uses dilation
            if self.kernel == 2:
                self._buffer = CausalQueue(self.dilation)
                conv_dilation = 1
            else:
                self._buffer = ShiftBuffer(self.receptive_field)
                conv_dilation = self.dilation
            
            self._conv = layers.Conv1D(
                self.channels, self.kernel,
                padding='valid',
                dilation_rate=conv_dilation,
                activation=self.activation,
                bias_initializer=self.bias_initializer,
                name='conv' if self.name else None
            )
        else:
            # Training mode: simple causal padding
            self._conv = layers.Conv1D(
                self.channels, self.kernel,
                padding='causal',
                dilation_rate=self.dilation,
                activation=self.activation,
                bias_initializer=self.bias_initializer,
                name='conv' if self.name else None
            )
        
        super(CausalConv1D, self).build(input_shape)
    
    def reset_states(self):
        if not self.stateful:
            raise RuntimeError("reset_states() can only be called when stateful=True")
        if self._buffer is not None:
            self._buffer.reset_states()
    
    def call(self, x):
        if self.stateful and self._buffer is not None:
            x = self._buffer(x)
        return self._conv(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'kernel': self.kernel,
            'dilation': self.dilation,
            'activation': self.activation,
            'stateful': self.stateful,
            'bias_initializer': self.bias_initializer,
        })
        return config


class PositionalEmbedding(layers.Layer):
    def __init__(self, width, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.width = width
    
    def build(self, input_shape = None):
        self.embedding = self.add_weight(name = 'embedding', 
                                         shape = (input_shape[1], self.width),
                                         trainable = True)
        
    def call(self, x):
        y = tf.tile(tf.expand_dims(self.embedding, 0), (tf.shape(x)[0], 1, 1))
        return tf.ensure_shape(y, x.shape[:2] + [self.width])
    
class LearnedGaussianNoise(layers.Layer):
    def __init__(self, sigma_initializer = 'ones', **kwargs):
        self.sigma_initializer = sigma_initializer
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        self.sigma = self.add_weight('sigma', shape = input_shape[-1], trainable = True, initializer = self.sigma_initializer)
        super().build(input_shape)

    def call(self, x):
        return x + tf.random.normal(tf.shape(x), dtype = self.compute_dtype) * self.sigma
    
class PrintLayer(layers.Layer):
    def call(self, x, *args):
        tf.print(*args)
        return x
    
class Resize1D(layers.Layer):
    def __init__(self, rate, method, **kwargs):
        self.rate = rate
        self.method = method
        super(Resize1D, self).__init__(**kwargs)
        
    def build(self, input_shape):
        super(Resize1D, self).build(input_shape)

    def call(self, x):
        x = tf.expand_dims(x, 2)
        w = tf.cast((x.shape[1] if x.shape[1] is not None else tf.shape(x)[1]) * self.rate, tf.int32)
        x = tf.image.resize(x, (w, 1), self.method)
        x = tf.squeeze(x, 2)
        return x
    
class TiedEmbedding(layers.Embedding):   
    def call(self, x, mode):
        if mode == 'input':
            return super().call(x)
        elif mode == 'output':
            return tf.matmul(x, self.embeddings, transpose_b = True)
        else:
            raise ValueError('Invalid mode %s' % mode)

class CausalTransformer(keras.layers.Layer):
    """
    Causal Transformer block with sliding window attention and ALiBi positional encoding.
    
    Supports both training and stateful inference modes:
    - Training (stateful=False): Processes full sequence with causal mask
    - Inference (stateful=True): Step-by-step with KV cache (sliding window)
    
    Uses ALiBi (Attention with Linear Biases) for position encoding, which:
    - Has no learned positional parameters
    - Naturally extrapolates to any sequence length
    - Just adds distance-based penalty to attention scores
    
    Args:
        embed_dim: Embedding dimension (must be divisible by num_heads)
        num_heads: Number of attention heads
        ff_dim: Feedforward hidden dimension (default: 4 * embed_dim)
        window_size: Sliding window size for attention (default: 512)
        dropout: Dropout rate (default: 0.0)
        stateful: Whether to use stateful KV caching for inference
        batch_size: Required if stateful=True
    """
    def __init__(self, embed_dim, num_heads, ff_dim=None, window_size=512, 
                 dropout=0.0, stateful=False, batch_size=None, **kwargs):
        super(CausalTransformer, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim or (4 * embed_dim)
        self.window_size = window_size
        self.dropout_rate = dropout
        self.stateful = stateful
        self.batch_size = batch_size
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        
        # Precompute ALiBi slopes (one per head)
        # Slopes are 2^(-8/n * i) for i in 1..n, where n = num_heads
        slopes = []
        for i in range(1, num_heads + 1):
            slopes.append(2.0 ** (-8.0 / num_heads * i))
        self._alibi_slopes = tf.constant(slopes, dtype=tf.float32)  # [num_heads]
        
    def build(self, input_shape):
        # QKV projections
        self.q_proj = layers.Dense(self.embed_dim, use_bias=False, name='q_proj')
        self.k_proj = layers.Dense(self.embed_dim, use_bias=False, name='k_proj')
        self.v_proj = layers.Dense(self.embed_dim, use_bias=False, name='v_proj')
        self.out_proj = layers.Dense(self.embed_dim, use_bias=False, name='out_proj')
        
        # Feedforward network
        self.ffn = keras.Sequential([
            layers.Dense(self.ff_dim, activation='gelu', name='ffn_up'),
            layers.Dense(self.embed_dim, name='ffn_down')
        ], name='ffn')
        
        # Layer norms (pre-norm architecture)
        self.ln1 = layers.LayerNormalization(epsilon=1e-6, name='ln1')
        self.ln2 = layers.LayerNormalization(epsilon=1e-6, name='ln2')
        
        # Dropout
        if self.dropout_rate > 0:
            self.dropout1 = layers.Dropout(self.dropout_rate)
            self.dropout2 = layers.Dropout(self.dropout_rate)
        
        # KV cache for stateful inference
        if self.stateful:
            if self.batch_size is None:
                raise ValueError("batch_size is required when stateful=True")
            
            # Cache shape: [batch, window_size, embed_dim]
            cache_shape = (self.batch_size, self.window_size, self.embed_dim)
            with tf.init_scope():
                k_cache = tf.Variable(
                    lambda: tf.zeros(cache_shape, dtype=self.compute_dtype),
                    trainable=False, name='k_cache'
                )
                v_cache = tf.Variable(
                    lambda: tf.zeros(cache_shape, dtype=self.compute_dtype),
                    trainable=False, name='v_cache'
                )
                cache_len = tf.Variable(0, trainable=False, dtype=tf.int32, name='cache_len')
            
            self._k_cache = self._no_dependency(k_cache)
            self._v_cache = self._no_dependency(v_cache)
            self._cache_len = self._no_dependency(cache_len)
        
        super(CausalTransformer, self).build(input_shape)
    
    def reset_states(self):
        if not self.stateful:
            raise RuntimeError("reset_states() can only be called when stateful=True")
        self._k_cache.assign(tf.zeros_like(self._k_cache))
        self._v_cache.assign(tf.zeros_like(self._v_cache))
        self._cache_len.assign(0)
    
    def _compute_alibi_bias(self, seq_len_q, seq_len_kv):
        """
        Compute ALiBi position bias matrix.
        
        ALiBi subtracts (slope * distance) from attention scores.
        For causal attention, we only penalize looking backward.
        
        Returns: [num_heads, seq_len_q, seq_len_kv]
        """
        # Position indices
        q_pos = tf.range(seq_len_q, dtype=tf.float32)  # [seq_len_q]
        kv_pos = tf.range(seq_len_kv, dtype=tf.float32)  # [seq_len_kv]
        
        # Distance matrix: for each query position, distance to each key position
        # Shape: [seq_len_q, seq_len_kv]
        # We want (q_pos - kv_pos) but only for positions where kv_pos <= q_pos (causal)
        # Offset q_pos by (seq_len_kv - seq_len_q) if using cached keys
        offset = seq_len_kv - seq_len_q
        q_pos_adjusted = q_pos + tf.cast(offset, tf.float32)
        
        distances = tf.expand_dims(q_pos_adjusted, 1) - tf.expand_dims(kv_pos, 0)  # [q, kv]
        distances = tf.maximum(distances, 0.0)  # Only look backward
        
        # Apply slopes per head: [num_heads, 1, 1] * [1, q, kv]
        slopes = tf.reshape(self._alibi_slopes, [self.num_heads, 1, 1])
        alibi = -slopes * tf.expand_dims(distances, 0)  # [num_heads, q, kv]
        
        return alibi
    
    def _attention(self, q, k, v, alibi_bias, causal_mask=None):
        """
        Multi-head attention with ALiBi.
        
        Args:
            q: [batch, seq_q, num_heads, head_dim]
            k: [batch, seq_kv, num_heads, head_dim]
            v: [batch, seq_kv, num_heads, head_dim]
            alibi_bias: [num_heads, seq_q, seq_kv]
            causal_mask: Optional [seq_q, seq_kv] boolean mask
        
        Returns: [batch, seq_q, embed_dim]
        """
        # Transpose to [batch, num_heads, seq, head_dim]
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])
        
        # Scaled dot-product attention
        scale = tf.math.rsqrt(tf.cast(self.head_dim, q.dtype))
        scores = tf.matmul(q, k, transpose_b=True) * scale  # [batch, heads, seq_q, seq_kv]
        
        # Add ALiBi bias
        alibi_bias = tf.cast(alibi_bias, scores.dtype)
        scores = scores + alibi_bias  # Broadcasting: [batch, heads, q, kv]
        
        # Apply causal mask if provided
        if causal_mask is not None:
            # causal_mask: [seq_q, seq_kv], True = attend, False = mask
            mask = tf.cast(~causal_mask, scores.dtype) * -1e9
            scores = scores + mask
        
        # Softmax and weighted sum
        attn_weights = tf.nn.softmax(scores, axis=-1)
        attn_output = tf.matmul(attn_weights, v)  # [batch, heads, seq_q, head_dim]
        
        # Transpose back and reshape
        attn_output = tf.transpose(attn_output, [0, 2, 1, 3])  # [batch, seq_q, heads, head_dim]
        batch_size = tf.shape(attn_output)[0]
        seq_len = tf.shape(attn_output)[1]
        attn_output = tf.reshape(attn_output, [batch_size, seq_len, self.embed_dim])
        
        return attn_output
    
    def call(self, x, training=False):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            training: Whether in training mode
            
        Returns: Output tensor [batch, seq_len, embed_dim]
        """
        if self.stateful:
            return self._call_stateful(x, training)
        else:
            return self._call_training(x, training)
    
    def _call_training(self, x, training):
        """Full sequence processing with causal mask."""
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        
        # Pre-norm
        x_norm = self.ln1(x)
        
        # QKV projections
        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)
        
        # Reshape to multi-head: [batch, seq, num_heads, head_dim]
        q = tf.reshape(q, [batch_size, seq_len, self.num_heads, self.head_dim])
        k = tf.reshape(k, [batch_size, seq_len, self.num_heads, self.head_dim])
        v = tf.reshape(v, [batch_size, seq_len, self.num_heads, self.head_dim])
        
        # Causal mask: lower triangular
        causal_mask = tf.linalg.band_part(tf.ones([seq_len, seq_len], dtype=tf.bool), -1, 0)
        
        # Sliding window mask (optional, for very long sequences during training)
        if self.window_size is not None:
            window_mask = tf.linalg.band_part(
                tf.ones([seq_len, seq_len], dtype=tf.bool), 
                self.window_size - 1, 0
            )
            causal_mask = causal_mask & window_mask
        
        # ALiBi bias
        alibi_bias = self._compute_alibi_bias(seq_len, seq_len)
        
        # Attention
        attn_out = self._attention(q, k, v, alibi_bias, causal_mask)
        attn_out = self.out_proj(attn_out)
        
        if self.dropout_rate > 0 and training:
            attn_out = self.dropout1(attn_out, training=training)
        
        # Residual
        x = x + attn_out
        
        # FFN with pre-norm
        x_norm = self.ln2(x)
        ffn_out = self.ffn(x_norm)
        
        if self.dropout_rate > 0 and training:
            ffn_out = self.dropout2(ffn_out, training=training)
        
        x = x + ffn_out
        
        return x
    
    def _call_stateful(self, x, training):
        """Step-by-step processing with KV cache."""
        # x: [batch, 1, embed_dim] for single-step inference
        batch_size = tf.shape(x)[0]
        
        # Pre-norm
        x_norm = self.ln1(x)
        
        # QKV projections for current step
        q = self.q_proj(x_norm)  # [batch, 1, embed_dim]
        k_new = self.k_proj(x_norm)
        v_new = self.v_proj(x_norm)
        
        # Update cache (sliding window)
        cache_len = self._cache_len
        
        # Shift cache left if full, then append new KV
        def shift_and_append(cache, new_val):
            # If cache is full, shift left by 1
            shifted = tf.concat([cache[:, 1:, :], new_val], axis=1)
            # If cache is not full, just append (overwrite at cache_len position)
            indices = tf.stack([
                tf.repeat(tf.range(self.batch_size), 1),
                tf.repeat(cache_len, self.batch_size)
            ], axis=1)
            updated = tf.tensor_scatter_nd_update(cache, indices, new_val[:, 0, :])
            return tf.cond(cache_len >= self.window_size, lambda: shifted, lambda: updated)
        
        new_k_cache = shift_and_append(self._k_cache, k_new)
        new_v_cache = shift_and_append(self._v_cache, v_new)
        
        self._k_cache.assign(new_k_cache)
        self._v_cache.assign(new_v_cache)
        self._cache_len.assign(tf.minimum(cache_len + 1, self.window_size))
        
        # Get valid cache entries
        valid_len = tf.minimum(cache_len + 1, self.window_size)
        k = self._k_cache[:, :valid_len, :]
        v = self._v_cache[:, :valid_len, :]
        
        # Reshape to multi-head
        q = tf.reshape(q, [batch_size, 1, self.num_heads, self.head_dim])
        k = tf.reshape(k, [batch_size, valid_len, self.num_heads, self.head_dim])
        v = tf.reshape(v, [batch_size, valid_len, self.num_heads, self.head_dim])
        
        # ALiBi bias (query attends to all cached keys)
        alibi_bias = self._compute_alibi_bias(1, valid_len)
        
        # Attention (no causal mask needed - we only have past + current in cache)
        attn_out = self._attention(q, k, v, alibi_bias, causal_mask=None)
        attn_out = self.out_proj(attn_out)
        
        # Residual
        x = x + attn_out
        
        # FFN with pre-norm
        x_norm = self.ln2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'window_size': self.window_size,
            'dropout': self.dropout_rate,
            'stateful': self.stateful,
            'batch_size': self.batch_size,
        })
        return config


class Codebook(layers.Layer):
    def __init__(self, num_codes, codes_initializer = 'random_normal', **kwargs):
        super().__init__(**kwargs)
        self.codes_initializer = codes_initializer
        self.num_codes = num_codes
        
    def build(self, input_shape):
        self.codes = self.add_weight(name = 'codes', 
                                     shape = (self.num_codes, input_shape[-1]),
                                     initializer = self.codes_initializer, 
                                     trainable = True)
            
        super().build(input_shape)
        
    def call(self, x, training=None):
        # training parameter accepted for Keras compatibility but not used
        x_f = tf.reshape(x, (-1, tf.shape(x)[-1]))
        c_t = tf.transpose(self.codes)
        d = tf.reduce_sum(tf.square(x_f), axis = 1, keepdims = True) - \
            2 * tf.matmul(x_f, c_t) + \
            tf.reduce_sum(tf.square(c_t), axis = 0, keepdims = True)
        i = tf.reshape(tf.argmin(d, axis = -1), tf.shape(x)[:-1])
        
        return tf.gather(self.codes, i), i
        
    def gather(self, i):
        return tf.gather(self.codes, i)
