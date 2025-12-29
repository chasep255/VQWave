import os
import random
import shlex
import subprocess

import numpy as np
import tensorflow as tf
import tinytag

def u16_to_f32(x):
    return np.float32((x - 32767.5) / 32767.5)

def f32_to_u16(x):
    return np.uint16(np.round(32767.5 * x + 32767.5))

def save_audio(f, r, x):
    x = f32_to_u16(x)
    ffmpeg_cmd = 'ffmpeg -hide_banner -loglevel warning -y -ar %d -ac 1 -channel_layout mono -f u16le -i pipe: -ac 1 %s' % (r, shlex.quote(f))
    with subprocess.Popen(shlex.split(ffmpeg_cmd), stdin = subprocess.PIPE) as p:
        p.stdin.write(x.tobytes())
        p.stdin.close()
        if p.wait():
            raise RuntimeError('ffmpeg failed to write audio data')
    
def load_audio(f, r, to_float = True):
    ffmpeg_cmd = 'ffmpeg -hide_banner -loglevel warning -channel_layout mono -i %s -ar %d -ac 1 -f u16le -c:a pcm_u16le pipe:' % (shlex.quote(f), r)
    with subprocess.Popen(shlex.split(ffmpeg_cmd), stdout = subprocess.PIPE) as p:
        buf = p.stdout.read()
        if p.wait():
            raise RuntimeError('ffmpeg failed to load audio data')
    x = np.frombuffer(buf, np.uint16)
    if to_float:
        x = u16_to_f32(x)
    return x

def load_meta(f):
    return tinytag.TinyTag.get(f).as_dict()

@tf.function(experimental_relax_shapes = True)
def mu_law(audio, mu = 255.0):
    mu = tf.cast(mu, audio.dtype)
    return tf.sign(audio) * tf.math.log(1 + mu * tf.abs(audio)) / tf.math.log(1.0 + mu)

@tf.function(experimental_relax_shapes = True)
def mu_law_inverse(audio, mu = 255.0):
    mu = tf.cast(mu, audio.dtype)
    return tf.sign(audio) * (1 / mu) * ((1 + mu) ** tf.abs(audio) - 1)

@tf.function(experimental_relax_shapes = True)
def mu_law_quantize(audio, quantization_channels = 256):
    mu = quantization_channels - 1
    return tf.cast((mu_law(audio, mu) + 1) / 2 * mu + 0.5, tf.int32)

@tf.function(experimental_relax_shapes = True)
def mu_law_dequantize(output, quantization_channels = 256):
    mu = quantization_channels - 1
    return mu_law_inverse(2 * (tf.cast(output, tf.float32) / mu) - 1, mu)
        
class AudioDataset:
    def __init__(self, path, min_length = 0, metadata = False, trim_start = 0.0, trim_end = 0.0):
        self.data = []
        self._shuffle_buf = None
        self.total_samples = 0
        self._trim_start = trim_start
        self._trim_end = trim_end
        for f in os.listdir(path):
            if not f.endswith('.u16'):
                continue
            x = np.memmap(os.path.join(path, f), dtype = np.uint16, mode = 'r')
            if x.shape[0] < min_length:
                continue
            md = {'file_artists': [x.strip() for x in f.split('-')[0].split(',')]}
            if metadata:
                with open(os.path.join(path, f[:-4]) + '.meta', 'r') as fd:
                    for l in fd:
                        l = l.strip()
                        if ':' in l:
                            l = l.split(':')
                            md[l[0]] = l[1]
            self.data.append((x, md))
            self.total_samples += self.data[-1][0].shape[0]
            
    def random_sample(self, length):
        if not self._shuffle_buf:
            self._shuffle_buf = list(self.data)
            random.shuffle(self._shuffle_buf)
    
        x, m = self._shuffle_buf.pop()
        s = int(self._trim_start * x.shape[0])
        e = x.shape[0] - length - int(x.shape[0] * self._trim_end)
        i = random.randint(s, e)
        return u16_to_f32(x[i : i + length]), m
    
    def random_batch(self, batch_size, sample_length):
        x, m = zip(*[self.random_sample(sample_length) for i in range(batch_size)])
        return np.float32(x), m
    
class AudioMix:
    def __init__(self, path, batch_size):
        self.data = []
        self.total_samples = 0
        for f in os.listdir(path):
            if not f.endswith('.u16'):
                continue
            x = np.memmap(os.path.join(path, f), dtype = np.uint16, mode = 'r')
            self.data.append(x)
            self.total_samples += x.shape[0]
        self.pos = [(random.choice(self.data), 0) for i in range(batch_size)]
    
    def next(self, sample_length):
        batch = []
        for i, (a, p) in enumerate(self.pos):
            s = a[p : p + sample_length]
            p += sample_length
            while s.shape[0] < sample_length:
                a = random.choice(self.data)
                p = 0
                l = sample_length - s.shape[0]
                s = np.append(s, a[p : p + l])
                p += l
            self.pos[i] = (a, p)
            batch.append(u16_to_f32(s))
        return np.float32(batch)

