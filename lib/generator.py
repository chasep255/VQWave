import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .layers import CausalQueue

def create_generator(stateful = False, batch_size = None):
    prior_code = keras.Input((None,), batch_size = batch_size, dtype = 'int32')
    
    # x = layers.Embedding(8192, 64)(prior_code)
    # x = layers.LSTM(1024, return_sequences = True, stateful = stateful)(x)
    # x = layers.LSTM(1024, return_sequences = True, stateful = stateful)(x)
    # x = layers.Conv1D(8192, 1, dtype = 'float32')(x)

    if stateful:
        xbuf = lambda x, d: CausalQueue(d)(x)
        xpad = 'valid'
    else:
        xbuf = lambda x, d: x
        xpad = 'causal'
    
    x = layers.Embedding(1024, 32 , name = 'input_emb')(prior_code)
    s = layers.Conv1D(512, 2, padding = xpad, name = 's_init')(xbuf(x, 1))
    x = layers.Conv1D(512, 2, padding = xpad, name = 'r_init')(xbuf(x, 1))
    for i, d in enumerate([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] * 2):
        x_skip = x
        x = layers.LayerNormalization(scale = False, center = False)(x)
        x = layers.Conv1D(256, 2, dilation_rate = 1 if stateful else d, padding = xpad, activation = 'swish', name = 'c_%d' % i)(xbuf(x, d))
        s += layers.Conv1D(512, 1, use_bias = False, name = 's_%d' % i)(x)
        x = x_skip + layers.Conv1D(512, 1, name = 'r_%d' % i)(x)
    
    x = layers.Concatenate(axis = -1)((
        layers.LayerNormalization(scale = False, center = False)(s),
        layers.LayerNormalization(scale = False, center = False)(x)
    ))
    x = layers.Conv1D(1024, 1, activation = 'swish', name = 'post_process_1')(x)
    x = layers.Conv1D(1024, 1, activation = 'swish', name = 'post_process_2')(x)
    x = layers.Conv1D(1024, 1, name = 'output', dtype = 'float32')(x)

    return keras.Model(inputs = prior_code, outputs = x, name = 'generator')

