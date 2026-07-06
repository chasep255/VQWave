"""
Diffusion decoder: renders a waveform from the VQ codes.

A fully convolutional, conditional denoising-diffusion model. It is the
high-fidelity "rendering" decoder -- the deterministic Decoder in encoder.py is
only the "training" decoder that shapes the codebook. This model learns
p(audio | z_q) and is used at generation time.

Architecture (see DIFFUSION_CONFIGS in config.py):
  - `encoder_layers` downsample the noisy waveform by exactly compression_rate to
    the code rate. The integer codes are embedded (a learned table) and the
    embeddings concatenated at that point; the trailing stride-1 encoder layers refine.
  - `decoder_layers` (transpose convs) expand back to the waveform.
  - Every conv is FiLM-conditioned on the diffusion time (AdaptiveShiftScale).
  - A final 1-channel float32 projection produces the (v / eps) prediction.

No attention or positional embeddings, so it runs on arbitrary-length audio
(any length that is a multiple of compression_rate).

Time convention (matching the WaveDiffuse / AudioDiffusion lineage): t=0 is pure
noise, t=1 is clean signal. x_t = signal_rate(t) * x + noise_rate(t) * eps.
Training uses a continuous-time cosine schedule with v-prediction; sampling is
deterministic DDIM.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input

from vqwave.config import DIFFUSION_CONFIGS, ENCODER_CONFIGS
from vqwave.audio import mu_law, mu_law_inverse


# --- Normalization: map the raw waveform marginal closer to a Gaussian so it
# matches the diffusion prior. A *mild* mu-law (mu=2.5) does this while lifting
# quiet detail; NORM_STD rescales to ~unit variance. These constants were tuned
# on the WaveDiffuse dataset -- re-derive them for a new dataset if needed.
NORM_MU = 2.5
NORM_STD = 0.3472


@tf.function(reduce_retracing=True)
def normalize(x):
    """Raw f32 waveform in [-1, 1] -> companded, ~unit-variance signal."""
    return mu_law(x, NORM_MU) / NORM_STD


@tf.function(reduce_retracing=True)
def denormalize(y):
    """Inverse of normalize(): model output -> f32 waveform in [-1, 1]."""
    return tf.clip_by_value(mu_law_inverse(y * NORM_STD, NORM_MU), -1.0, 1.0)


class AdaptiveShiftScale(layers.Layer):
    """FiLM / adaLN time conditioning: modulate x by (gamma, beta) from t.

    gamma starts near 1 (sigmoid bias 2) and beta near 0 so the layer begins as a
    near-identity, which stabilises early training. t is one vector per batch
    element, broadcast over the time axis.
    """

    def build(self, input_shape):
        self.gamma = layers.Dense(input_shape[-1], activation='sigmoid',
                                  kernel_initializer='zeros',
                                  bias_initializer=keras.initializers.Constant(2))
        self.beta = layers.Dense(input_shape[-1], kernel_initializer='zeros',
                                 bias_initializer='zeros')
        super().build(input_shape)

    def call(self, x, t):
        beta = self.beta(t)
        gamma = self.gamma(t)
        s = [tf.shape(x)[0]] + [1] * (len(x.shape) - 2) + [-1]
        beta = tf.reshape(beta, s)
        gamma = tf.reshape(gamma, s)
        return x * gamma + beta


class TimeEmbedding(layers.Layer):
    """Map a diffusion time scalar t in [0, 1] to a conditioning vector.

    Fixed log-spaced sinusoidal (Fourier) features followed by a small MLP.
    """

    def __init__(self, dim, num_freqs=64, activation='swish', **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_freqs = num_freqs
        freqs = np.exp(np.linspace(0.0, np.log(1000.0), num_freqs)).astype(np.float32)
        self.freqs = tf.constant(freqs * np.pi, dtype=tf.float32)
        self.dense1 = layers.Dense(dim, activation=activation)
        self.dense2 = layers.Dense(dim, activation=activation)

    def call(self, t):
        t = tf.cast(tf.reshape(t, (-1, 1)), tf.float32)      # (B, 1)
        ang = t * self.freqs                                  # (B, num_freqs)
        eps = 1e-4
        emb = tf.concat([
            t,
            tf.sin(ang), tf.cos(ang),
            tf.math.log(t + eps) / 9.0,
            tf.math.log(1.0 - t + eps) / 9.0,
        ], axis=-1)
        return self.dense2(self.dense1(emb))                  # (B, dim)


def _film_conv(x, t, channels, kernel, stride, activation, transpose=False):
    """One time-FiLM-conditioned conv (straight, strided, or transposed)."""
    Conv = layers.Conv1DTranspose if transpose else layers.Conv1D
    x = Conv(channels, kernel, strides=stride, padding='same', use_bias=False)(x)
    x = AdaptiveShiftScale()(x, t)
    if activation:
        x = layers.Activation(activation)(x)
    return x


class Denoiser(keras.Model):
    """Conditional convolutional waveform denoiser for the diffusion decoder.

    Inputs:  (audio (B, L) normalized waveform, time (B,) in [0, 1],
              codes  (B, L/compression) integer VQ code ids)
    Output:  prediction (B, L) -- v-target (default) or noise eps.
    """

    def __init__(self, config="diffusion_256", **kwargs):
        if isinstance(config, str):
            config = DIFFUSION_CONFIGS[config]

        vqvae = ENCODER_CONFIGS[config["dest_vqvae"]]
        compression = vqvae["compression_rate"]
        num_codes = vqvae["num_codes"]

        enc_layers = config["encoder_layers"]
        dec_layers = config["decoder_layers"]

        # The encoder must downsample by exactly compression_rate, and the decoder
        # (transpose convs) must expand back by the same factor.
        down = 1
        for l in enc_layers:
            down *= l.get("stride", 1)
        assert down == compression, (
            "encoder_layers downsample by %d but dest VQ-VAE compresses by %d"
            % (down, compression))
        up = 1
        for l in dec_layers:
            if l.get("transpose", False):
                up *= l.get("stride", 1)
        assert up == compression, (
            "decoder_layers expand by %d but dest VQ-VAE compresses by %d"
            % (up, compression))

        if 'name' not in kwargs:
            kwargs['name'] = config["dest_vqvae"].replace("vqvae", "diffusion")

        input_audio = Input((None,), name='audio')
        input_time = Input((), name='time')
        input_codes = Input((None,), dtype='int32', name='codes')

        t = TimeEmbedding(config["time_dim"])(input_time)
        zc = layers.Embedding(num_codes, config["cond_dim"], name='code_embedding')(input_codes)

        # --- Encoder: downsample to the code rate, concat codes, then refine ---
        # A skip is kept at each resolution (last layer at that resolution wins) so
        # the decoder can recover the high-frequency detail of the noisy input --
        # without skips it would all have to squeeze through the bottleneck.
        x = layers.Reshape((-1, 1))(input_audio)
        skips = {}
        cum = 1
        injected = False
        for l in enc_layers:
            stride = l.get("stride", 1)
            x = _film_conv(x, t, l["channels"], l["kernel"], stride,
                           l.get("activation", "elu"))
            cum *= stride
            if not injected and cum == compression:
                # At the code rate: fuse the code embeddings.
                x = layers.Concatenate(axis=-1, name='code_concat')((x, zc))
                injected = True
            skips[cum] = x
        assert injected, "encoder never reached the code rate to inject codes"

        # --- Decoder: expand back to the waveform, concatenating encoder skips at
        # matching resolutions right before each upsample. ---
        cum = compression
        for l in dec_layers:
            transpose = l.get("transpose", False)
            stride = l.get("stride", 1)
            if transpose and cum != compression and cum in skips:
                x = layers.Concatenate(axis=-1)((x, skips[cum]))
            x = _film_conv(x, t, l["channels"], l["kernel"], stride,
                           l.get("activation", "elu"), transpose=transpose)
            if transpose:
                cum //= stride

        # Final projection to a single-channel prediction (linear, float32).
        x = layers.Conv1D(1, 1, dtype='float32', name='output_proj')(x)
        x = layers.Flatten(dtype='float32')(x)

        super().__init__(inputs=(input_audio, input_time, input_codes),
                         outputs=x, **kwargs)
        self.compression = compression
        self.num_codes = num_codes
        self.prediction = config["prediction"]

    # ----- Continuous-time cosine (angular) schedule -----
    @tf.function
    def signal_rate(self, t):
        return tf.sin(0.5 * np.pi * (0.98 * t + 0.01))

    @tf.function
    def noise_rate(self, t):
        return tf.cos(0.5 * np.pi * (0.98 * t + 0.01))

    def merge_noise(self, x, noise, t):
        t = tf.reshape(t, (-1, 1))
        sr = tf.cast(self.signal_rate(t), x.dtype)
        nr = tf.cast(self.noise_rate(t), noise.dtype)
        return sr * x + nr * noise

    def target(self, x, noise, t):
        """Regression target for the network given clean x and noise at time t."""
        if self.prediction == 'eps':
            return noise
        t = tf.reshape(t, (-1, 1))
        sr = tf.cast(self.signal_rate(t), x.dtype)
        nr = tf.cast(self.noise_rate(t), x.dtype)
        return sr * noise - nr * x        # v-target

    def split_prediction(self, x_t, pred, t):
        """Recover (x0_hat, eps_hat) from the network prediction at time t."""
        t = tf.reshape(t, (-1, 1))
        sr = tf.cast(self.signal_rate(t), x_t.dtype)
        nr = tf.cast(self.noise_rate(t), x_t.dtype)
        if self.prediction == 'eps':
            eps = pred
            x0 = (x_t - nr * eps) / sr
        else:
            x0 = sr * x_t - nr * pred
            eps = nr * x_t + sr * pred
        return tf.clip_by_value(x0, -3.0, 3.0), eps

    def generate(self, codes, nsteps=200, eta=0.0, initial_noise=None, progress=False):
        """DDIM sampling from noise (t=0) to signal (t=1), conditioned on codes.

        codes: integer VQ code ids of shape (B, T); the output length is
        T * compression. eta controls stochasticity: 0 = deterministic DDIM, up to
        1 = fully ancestral. Returns normalized waveforms (B, L); call denormalize()
        to get f32 audio in [-1, 1].
        """
        codes = tf.cast(tf.convert_to_tensor(codes), tf.int32)
        batch = tf.shape(codes)[0]
        length = tf.shape(codes)[1] * self.compression
        x_t = tf.random.normal((batch, length)) if initial_noise is None else initial_noise
        times = tf.cast(tf.linspace(0.0, 1.0, nsteps + 1), tf.float32)
        x0 = x_t
        for i in range(nsteps):
            t = tf.fill((batch,), times[i])
            pred = self((x_t, t, codes), training=False)
            x0, eps = self.split_prediction(x_t, pred, t)
            if eta > 0.0:
                eps = eta * tf.random.normal(tf.shape(eps)) + tf.sqrt(1.0 - eta ** 2) * eps
            t1 = tf.fill((batch,), times[i + 1])
            x_t = self.merge_noise(x0, eps, t1)
            if progress:
                print('  sampling step %d/%d' % (i + 1, nsteps), end='\r')
        if progress:
            print()
        return x0
