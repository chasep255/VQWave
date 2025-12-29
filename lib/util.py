import numpy as np
import tensorflow as tf

class AverageAccumulator:
    """Accumulate and compute average of values."""
    def __init__(self):
        self.count = 0.0
        self.sum = 0.0
    
    def add(self, x, w=1.0):
        if np.isfinite(x):
            self.count += w
            self.sum += x * w
    
    def get(self):
        return self.sum / self.count if self.count else 0.0


class LRWarmupWrapper(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr, growth_rate = 1e-4, initial_step = 0):
        self.lr = lr
        self.growth_rate = growth_rate
        self.initial_step = initial_step
    
    def __call__(self, step):
        r = tf.clip_by_value(self.growth_rate * tf.cast(step - self.initial_step, tf.float32), 0, 1)
        if callable(self.lr):
            return r * self.lr(step)
        else:
            return r * self.lr
    
class GradientAccumulator:
    def __init__(self, weights):
        self.grads = [tf.Variable(tf.zeros_like(w)) for w in weights]
        self.counter = tf.Variable(0, dtype = tf.int32)
    
    @tf.function
    def accumulate(self, grads):
        for g, g_ in zip(self.grads, grads):
            g.assign_add(g_)
        self.counter.assign_add(1)
        
    @tf.function
    def gradients(self):
        return [g / tf.cast(self.counter, g.dtype) for g in self.grads]
    
    @tf.function
    def clear(self):
        self.counter.assign(0)
        for g in self.grads:
            g.assign(tf.zeros_like(g))
            
class CodebookRestarter:
    def __init__(self, codebook, limit, random_init = False):
        self._limit = limit
        self._codebook = codebook
        if random_init:
            self._code_counter = tf.Variable(tf.random.uniform([self._codebook.num_codes], 0, self._limit, dtype = tf.int32), trainable = False)
        else:
            self._code_counter = tf.Variable(self._limit + tf.zeros([self._codebook.num_codes], tf.int32), trainable = False)
    
    @tf.function
    def update(self, x, i):
        u = tf.unique(tf.reshape(i, [-1]))[0]
        self._code_counter.assign(tf.tensor_scatter_nd_update(self._code_counter - 1, tf.expand_dims(u, axis = -1), tf.fill(tf.shape(u), self._limit)))
        unused = self._code_counter <= 0
        num_unused = tf.reduce_sum(tf.cast(unused, tf.int32))
        if num_unused > 0:
            x_= tf.reshape(x, (-1, tf.shape(x)[-1]))
            new_codes = tf.gather(x_, tf.random.uniform([num_unused], maxval = tf.shape(x_)[-1], dtype = tf.int32))
            self._codebook.codes.assign(tf.tensor_scatter_nd_update(self._codebook.codes, tf.where(unused), new_codes))
            self._code_counter.assign(tf.where(unused, self._limit, self._code_counter))
        return u, num_unused

