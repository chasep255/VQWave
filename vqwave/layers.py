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
        self._queue = tf.queue.FIFOQueue(self._dilation_rate, tf.float32, shapes = [1 if x is None else x for x in input_shape])
        super(CausalQueue, self).build(input_shape)
        
    def reset_states(self):
        self._queue.dequeue_many(self._queue.size())
        
    def call(self, x):
        x_past = tf.cond(self._dilation_rate == self._queue.size(), lambda: self._queue.dequeue(), lambda: tf.zeros_like(x))
        x_past = tf.ensure_shape(x_past, x.shape)
        self._queue.enqueue(x)
        return tf.concat((x_past, x), 1)
    
class ShiftBuffer(keras.layers.Layer):
    def __init__(self, width):
        super(ShiftBuffer, self).__init__()
        self.width = width
        
    def build(self, input_shape):
        self.buffer = tf.Variable(tf.zeros((input_shape[0], self.width, input_shape[2])))
        super(ShiftBuffer, self).build(input_shape)
        
    def reset_states(self):
        self.buffer.assign(tf.zeros_like(self.buffer))
        
    def call(self, x):
        self.buffer.assign(tf.concat((self.buffer[:, 1:], x), 1))
        return self.buffer
    
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
