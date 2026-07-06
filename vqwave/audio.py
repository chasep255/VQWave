import os
import queue
import random
import shlex
import subprocess
import threading

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
    """Random-crop dataset over .u16 raw audio files.

    Stores only file paths and lengths (computed from file size, not by opening
    or memory-mapping every file), so it scales to tens of thousands of files
    without exhausting the open-file limit. Each crop opens its file on demand,
    reads only the requested slice, and closes immediately. Returns raw f32
    waveforms in [-1, 1].
    """

    def __init__(self, path, min_length=0, trim_start=0.0, trim_end=0.0):
        self.data = []                       # list of (filepath, num_samples)
        self.total_samples = 0
        self._trim_start = trim_start
        self._trim_end = trim_end

        for f in os.listdir(path):
            if not f.endswith('.u16'):
                continue
            fp = os.path.join(path, f)
            n = os.path.getsize(fp) // 2     # uint16 -> 2 bytes per sample
            if n < max(min_length, 1):
                continue
            self.data.append((fp, n))
            self.total_samples += n

        if not self.data:
            raise ValueError('No .u16 files found in %s' % path)

    def random_sample(self, length):
        # Stateless (no shared shuffle buffer) so it is safe to call from many
        # loader threads at once; each call opens and reads its own crop.
        while True:
            fp, n = random.choice(self.data)
            s = int(self._trim_start * n)
            e = n - length - int(n * self._trim_end)
            if e < s:                                       # too short for this crop length
                continue

            i = random.randint(s, e)
            with open(fp, 'rb') as fd:                      # read only the crop, not the whole file
                fd.seek(i * 2)                              # uint16 -> 2 bytes per sample
                buf = fd.read(length * 2)
            return u16_to_f32(np.frombuffer(buf, dtype=np.uint16))

    def random_batch(self, batch_size, sample_length):
        return np.float32([self.random_sample(sample_length) for _ in range(batch_size)])


class AudioLoader:
    """Background-threaded prefetcher around AudioDataset.

    A single daemon worker thread continuously builds random batches and pushes
    them onto a bounded queue, so the training loop pulls a ready batch each step
    instead of blocking on disk I/O -- loading overlaps with GPU compute. The
    daemon thread exits with the process.
    """

    def __init__(self, path, length, batch_size, min_length=None,
                 queue_size=8, trim_start=0.0, trim_end=0.0):
        self.dataset = AudioDataset(path, min_length=min_length or length,
                                    trim_start=trim_start, trim_end=trim_end)
        self._length = length
        self._batch_size = batch_size
        self._queue = queue.Queue(maxsize=queue_size)
        threading.Thread(target=self._worker, daemon=True).start()

    @property
    def data(self):
        return self.dataset.data

    @property
    def total_samples(self):
        return self.dataset.total_samples

    def _worker(self):
        while True:
            self._queue.put(self.dataset.random_batch(self._batch_size, self._length))

    def random_batch(self):
        """Block only if no prefetched batch is ready (i.e. when I/O-bound)."""
        return self._queue.get()


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

