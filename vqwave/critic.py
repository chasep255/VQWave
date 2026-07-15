"""
WGAN-GP critic for adversarial VQ-VAE training.

The critic scores a waveform's realism. It is used only during training: the
VQ-VAE decoder is pushed to produce reconstructions the critic cannot separate
from real audio, on top of the usual reconstruction + commitment losses.

Notes specific to WGAN-GP:
  - The output is an unbounded scalar (no sigmoid). The critic estimates the
    Wasserstein distance, it does not classify.
  - There are NO normalization layers. BatchNorm would couple examples within a
    batch, which invalidates the per-example gradient penalty. (LayerNorm would
    be acceptable; we simply use none.)
"""

import tensorflow as tf
from tensorflow.keras import layers, Input, Model


class Critic(Model):
    """Waveform critic: [batch, samples] -> [batch, 1] unbounded score."""

    def __init__(self, config, name='critic', **kwargs):
        """
        Args:
            config: VQ-VAE config dict containing a "critic" entry:
                - layers: list of {channels, kernel, stride, alpha}
        """
        critic_layers = config["critic"]["layers"]

        input_audio = Input((None,), name='audio_input')
        x = layers.Reshape((-1, 1))(input_audio)

        for i, lc in enumerate(critic_layers):
            x = layers.Conv1D(
                lc["channels"], lc["kernel"],
                strides=lc.get("stride", 1),
                padding='same',
                name=f'critic_conv_{i}',
            )(x)
            x = layers.LeakyReLU(
                negative_slope=lc.get("alpha", 0.2), name=f'critic_act_{i}'
            )(x)

        # Per-window scores, averaged into one score per example. float32 so the
        # score and the gradient penalty stay full precision under bf16.
        x = layers.Conv1D(1, 3, padding='same', dtype='float32', name='critic_out')(x)
        score = layers.GlobalAveragePooling1D(dtype='float32', name='critic_score')(x)

        super().__init__(inputs=input_audio, outputs=score, name=name, **kwargs)
        self.config = config


def gradient_penalty(critic, real, fake):
    """WGAN-GP penalty: E[(||d D(x_hat) / d x_hat|| - 1)^2] on interpolates.

    x_hat is sampled uniformly on the line between each real and fake example.
    """
    batch = tf.shape(real)[0]
    eps = tf.random.uniform((batch, 1), 0.0, 1.0, dtype=real.dtype)
    x_hat = eps * real + (1.0 - eps) * fake

    with tf.GradientTape() as tape:
        tape.watch(x_hat)
        score = critic(x_hat, training=True)
    grads = tape.gradient(score, x_hat)                      # [batch, samples]

    # Epsilon inside the sqrt: the norm's gradient is singular at 0, which would
    # produce NaNs the moment an interpolate lands on a flat/zero gradient.
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-12)
    return tf.reduce_mean(tf.square(norm - 1.0))
