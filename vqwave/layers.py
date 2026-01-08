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
