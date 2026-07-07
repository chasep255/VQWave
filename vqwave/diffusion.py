"""
Diffusion decoder: refines the deterministic VQ-VAE reconstruction.

A fully convolutional, conditional denoising-diffusion model that sharpens the
deterministic VQ-VAE decoder's output. The decoded waveform is concatenated
channel-wise with the noisy input, so the conditioning is present at full
resolution and propagates through every level via the skip connections -- much
stronger guidance than injecting codes at the coarse bottleneck.

Architecture (WaveDiffuse-style U-Net; see DIFFUSION_CONFIGS in config.py):
  - The noisy waveform and the decoded-waveform conditioning are concatenated into
    a 2-channel input.
  - Contracting `stages` downsample it. Each level is two convs (strided down +
    stride-1 refine) and yields one skip.
  - The `middle` projects into a residual stream, then a stack of pre-activated
    residual dilated blocks widens the receptive field.
  - Expanding stages mirror the contracting path: concat the level's skip, fuse,
    then upsample with a transposed conv.
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


def _film_conv(x, t, channels, kernel, stride=1, activation='elu',
               transpose=False, dilation=1):
    """One time-FiLM-conditioned conv (straight, strided, dilated, or transposed)."""
    if transpose:
        x = layers.Conv1DTranspose(channels, kernel, strides=stride,
                                   padding='same', use_bias=False)(x)
    else:
        x = layers.Conv1D(channels, kernel, strides=stride, dilation_rate=dilation,
                          padding='same', use_bias=False)(x)
    x = AdaptiveShiftScale()(x, t)
    if activation:
        x = layers.Activation(activation)(x)
    return x


def _res_block(x, t, channels, kernel, dilation):
    """Pre-activated residual block with two convs:

        h = Conv_dilated(ELU(FiLM(x, t)))    # mix over time (kernel, dilation)
        h = Conv_1x1(ELU(FiLM(h, t)))        # reproject over channels (dense in time)
        return x + h

    Pre-activation (He et al. 2016) keeps the residual stream un-activated, which
    trains better through a deep stack; FiLM injects the diffusion time before
    each conv. The identity add requires matching channels (guaranteed by the
    middle input projection); otherwise the block is a plain projection."""
    h = AdaptiveShiftScale()(x, t)
    h = layers.Activation('elu')(h)
    h = layers.Conv1D(channels, kernel, dilation_rate=dilation,
                      padding='same', use_bias=False)(h)          # dilated conv over time
    h = AdaptiveShiftScale()(h, t)
    h = layers.Activation('elu')(h)
    h = layers.Conv1D(channels, 1, padding='same', use_bias=False)(h)  # 1x1 reproject
    if x.shape[-1] == channels:
        return layers.Add()((x, h))
    return h


class Denoiser(keras.Model):
    """Conditional convolutional waveform denoiser for the diffusion decoder.

    Conditioned on the deterministic VQ-VAE reconstruction: the decoded waveform
    is concatenated channel-wise with the noisy input, so the guidance is present
    at full resolution and flows through every level (via the skips). The model
    acts as a refiner on top of the deterministic decoder.

    Inputs:  (audio (B, L) normalized noisy waveform, time (B,) in [0, 1],
              decoded (B, L) normalized deterministic VQ-VAE reconstruction)
    Output:  prediction (B, L) -- v-target (default) or noise eps.
    """

    def __init__(self, config="diffusion_256", **kwargs):
        if isinstance(config, str):
            config = DIFFUSION_CONFIGS[config]

        vqvae = ENCODER_CONFIGS[config["dest_vqvae"]]
        compression = vqvae["compression_rate"]

        stages = config["stages"]
        middle = config["middle"]

        # The U-Net downsamples by the product of the stage strides; keeping this
        # equal to compression_rate makes valid input lengths (multiples of the
        # VQ-VAE compression) line up with the U-Net's down/up symmetry.
        down = 1
        for s in stages:
            down *= s["stride"]
        assert down == compression, (
            "stage strides multiply to %d but dest VQ-VAE compresses by %d"
            % (down, compression))

        if 'name' not in kwargs:
            kwargs['name'] = config["dest_vqvae"].replace("vqvae", "diffusion")

        input_audio = Input((None,), name='audio')
        input_time = Input((), name='time')
        input_decoded = Input((None,), name='decoded')

        t = TimeEmbedding(config["time_dim"])(input_time)

        # --- Concatenate the noisy input with the decoded-waveform conditioning
        # into a 2-channel signal, then run the U-Net. ---
        x = layers.Concatenate(axis=-1, name='cond_concat')((
            layers.Reshape((-1, 1))(input_audio),
            layers.Reshape((-1, 1))(input_decoded),
        ))

        # --- Contracting path: two convs per level (strided down + refine), one
        # skip per level. Skips let the decoder recover high-frequency detail. ---
        skips = []
        for s in stages:
            x = _film_conv(x, t, s["channels"], s["resample_kernel"], stride=s["stride"])  # downsample
            x = _film_conv(x, t, s["channels"], s["conv_kernel"])                           # refine
            skips.append(x)

        # --- Middle: project into the residual stream, then a stack of
        # pre-activated residual dilated blocks. ---
        x = _film_conv(x, t, middle[0]["channels"], 1, activation=None)  # project to residual width
        for m in middle:
            x = _res_block(x, t, m["channels"], m["kernel"], m.get("dilation", 1))
        x = layers.Activation('elu')(x)  # final activation out of the residual stream

        # --- Expanding path: mirror of the contracting path. Concat the level's
        # skip, fuse, then upsample with a transposed conv. ---
        for i in reversed(range(len(stages))):
            s = stages[i]
            x = layers.Concatenate(axis=-1)((x, skips[i]))
            x = _film_conv(x, t, s["channels"], s["conv_kernel"])                # fuse skip
            out_c = stages[i - 1]["channels"] if i > 0 else s["channels"]
            x = _film_conv(x, t, out_c, s["resample_kernel"], stride=s["stride"], transpose=True)

        # Final projection to a single-channel prediction (linear, float32).
        x = layers.Conv1D(1, 1, dtype='float32', name='output_proj')(x)
        x = layers.Flatten(dtype='float32')(x)

        super().__init__(inputs=(input_audio, input_time, input_decoded),
                         outputs=x, **kwargs)
        self.compression = compression
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

    def generate(self, decoded, nsteps=200, eta=0.0, initial_noise=None, progress=False):
        """DDIM sampling from noise (t=0) to signal (t=1), conditioned on the
        deterministic VQ-VAE reconstruction.

        decoded: normalized deterministic reconstruction of shape (B, L) (the
        output has the same length). eta controls stochasticity: 0 = deterministic
        DDIM, up to 1 = fully ancestral. Returns normalized waveforms (B, L); call
        denormalize() to get f32 audio in [-1, 1].
        """
        decoded = tf.convert_to_tensor(decoded)
        batch = tf.shape(decoded)[0]
        length = tf.shape(decoded)[1]
        x_t = tf.random.normal((batch, length)) if initial_noise is None else initial_noise
        times = tf.cast(tf.linspace(0.0, 1.0, nsteps + 1), tf.float32)
        x0 = x_t
        for i in range(nsteps):
            t = tf.fill((batch,), times[i])
            pred = self((x_t, t, decoded), training=False)
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
