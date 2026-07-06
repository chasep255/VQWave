#!/usr/bin/env python3
"""
Train a VQ-VAE (vqvae_256 / vqvae_512) with a reconstruction loss.

The encoder, decoder, and codebook are trained with a reconstruction loss
(STFT / mel / MSE) plus a VQ commitment loss. The deterministic decoder trained
here shapes the codebook; the diffusion decoder (scripts/train_diffusion.py)
renders higher-fidelity audio from the codes at generation time.
"""

import argparse
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

from vqwave.encoder import Encoder, Decoder, CodebookManager
from vqwave.config import ENCODER_CONFIGS, SAMPLE_RATE
from vqwave.audio import AudioLoader
from vqwave.util import AverageAccumulator, CodebookRestarter


# GPU setup
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


@tf.function
def stft_loss(y, r):
    loss = []
    for w, s in ((220, 41), (770, 109), (1100, 239)):
        y_ = tf.abs(tf.signal.stft(y, w, s))
        r_ = tf.abs(tf.signal.stft(r, w, s))
        loss.append(tf.reduce_mean(tf.abs(y_ - r_)))
    return tf.reduce_mean(loss)


@tf.function
def mel_loss(y, r, sample_rate=SAMPLE_RATE, n_mels=80, fft_length=2048, hop_length=512):
    """Compute mel spectrogram loss (not log mel)."""
    y_stft = tf.signal.stft(y, fft_length, hop_length, fft_length, pad_end=True)
    r_stft = tf.signal.stft(r, fft_length, hop_length, fft_length, pad_end=True)

    y_mag = tf.abs(y_stft)
    r_mag = tf.abs(r_stft)

    num_spectrogram_bins = fft_length // 2 + 1
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=sample_rate,
        lower_edge_hertz=0.0,
        upper_edge_hertz=sample_rate / 2.0
    )

    y_mel = tf.tensordot(y_mag, mel_matrix, axes=1)
    r_mel = tf.tensordot(r_mag, mel_matrix, axes=1)
    return tf.reduce_mean(tf.abs(y_mel - r_mel))


@tf.function
def mse_loss(y, r):
    """Compute simple mean squared error loss on waveforms."""
    return tf.reduce_mean(tf.square(y - r))


# Loss function registry
LOSS_FUNCTIONS = {
    'stft': stft_loss,
    'mel': mel_loss,
    'mse': mse_loss
}


@tf.function
def train_step(encoder, decoder, codebook,
               optimizer, restarter, r, loss_fn, commit_weight):
    """
    Single training step.

    Updates the autoencoder (encoder, decoder, codebook) with reconstruction +
    commitment losses. The commit weight is passed as a float tensor to avoid
    retracing when it is tuned.
    """
    with tf.GradientTape() as tape:
        z_e = encoder(r, training=True)
        z_q, codes = codebook(z_e, training=True)

        # Straight-through estimator: forward uses z_q, backward passes through z_e
        z_q_st = z_e + tf.stop_gradient(z_q - z_e)

        y = decoder(z_q_st, training=True)

        audio_loss = loss_fn(y, r)
        commit_loss = commit_weight * tf.reduce_mean(tf.square(z_e - z_q))

        loss = audio_loss + commit_loss

    weights = (decoder.trainable_weights +
               codebook.trainable_weights +
               encoder.trainable_weights)
    grads = tape.gradient(loss, weights)
    optimizer.apply_gradients(zip(grads, weights))

    num_used, num_reset = restarter.update(z_e, codes)
    num_used = tf.shape(num_used)[0]

    return {
        'loss': loss,
        'audio_loss': audio_loss,
        'commit_loss': commit_loss,
        'used': num_used,
        'reset': num_reset,
    }


def main():
    parser = argparse.ArgumentParser(description='Train the 256x VQ-VAE with a reconstruction loss')
    parser.add_argument('--model', type=str, required=True,
                       choices=list(ENCODER_CONFIGS.keys()),
                       help=f'Model preset name (choices: {", ".join(ENCODER_CONFIGS.keys())})')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing training audio files')
    parser.add_argument('--output-dir', type=str, default='weights',
                       help='Directory to save model weights (default: weights)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size (default: 8)')
    parser.add_argument('--start-epoch', type=int, default=0,
                       help='Starting epoch number (default: 0)')
    parser.add_argument('--load-weights', action='store_true', default=False,
                       help='Load existing weights from output directory (default: False)')
    parser.add_argument('--warmup-steps', type=int, default=0,
                       help='Adam moment warmup: run N steps to settle the optimizer '
                            'moments (m, v) with weights frozen before training (default: 0, no warmup)')
    parser.add_argument('--input-length', type=int, default=2**16,
                       help='Input audio length in samples (default: 65536)')
    parser.add_argument('--epoch-steps', '--steps', type=int, default=10000,
                       help='Number of training steps per epoch (default: 10000)')
    parser.add_argument('--code-reset-limit', type=int, default=256,
                       help='Codebook reset limit (default: 256)')
    parser.add_argument('--learning-rate', '--lr', type=float, default=1e-4,
                       help='Initial learning rate (default: 1e-4)')
    parser.add_argument('--decay-rate', type=float, default=0.9,
                       help='Multiplicative LR decay applied every --decay-steps (default: 0.9)')
    parser.add_argument('--decay-steps', type=int, default=None,
                       help='Number of steps for each decay (default: epoch_steps)')
    parser.add_argument('--loss', type=str, default='stft',
                       choices=list(LOSS_FUNCTIONS.keys()),
                       help=f'Reconstruction loss (default: stft, choices: {", ".join(LOSS_FUNCTIONS.keys())})')
    parser.add_argument('--bf16', action='store_true', default=False,
                       help='Use mixed_bfloat16 precision (Ampere+; half memory, no loss scaling)')
    parser.add_argument('--commit-weight', type=float, default=0.01,
                       help='Weight on the VQ commitment loss (default: 0.01)')

    args = parser.parse_args()

    # Mixed precision must be set before models are built.
    if args.bf16:
        keras.mixed_precision.set_global_policy('mixed_bfloat16')
        print("Using mixed_bfloat16 precision")

    config = ENCODER_CONFIGS[args.model]
    encoder = Encoder(config)
    decoder = Decoder(config)
    codebook = CodebookManager(config)

    print("Encoder:")
    encoder.summary()
    print("\nDecoder:")
    decoder.summary()
    print("\nCodebook:")
    codebook.summary()

    loss_fn = LOSS_FUNCTIONS[args.loss]
    loss_name = args.loss.upper()
    print(f"\nUsing {loss_name} reconstruction loss")
    print(f"Loss weights: commit={args.commit_weight}")

    # Load weights if resuming training or --load-weights flag is set
    if args.start_epoch > 0 or args.load_weights:
        model_prefix = args.model
        encoder_path = os.path.join(args.output_dir, f'{model_prefix}_encoder.weights.h5')
        decoder_path = os.path.join(args.output_dir, f'{model_prefix}_decoder.weights.h5')
        codebook_path = os.path.join(args.output_dir, f'{model_prefix}_codebook.weights.h5')

        if os.path.exists(encoder_path) and os.path.exists(decoder_path) and os.path.exists(codebook_path):
            print(f"\nLoading weights from {args.output_dir}...")
            encoder.load_weights(encoder_path)
            decoder.load_weights(decoder_path)
            codebook.load_weights(codebook_path)
            print("Weights loaded successfully.")
        else:
            missing = []
            if not os.path.exists(encoder_path):
                missing.append('encoder')
            if not os.path.exists(decoder_path):
                missing.append('decoder')
            if not os.path.exists(codebook_path):
                missing.append('codebook')
            raise FileNotFoundError(
                f"Weight files not found in {args.output_dir}. Missing: {', '.join(missing)}. "
                f"Expected: {encoder_path}, {decoder_path}, {codebook_path}"
            )

    # Load dataset (background-threaded prefetcher; reads crops on demand)
    loader = AudioLoader(args.data_dir, args.input_length, args.batch_size)
    secs = loader.total_samples / SAMPLE_RATE
    print('%02d:%02d:%02d of training audio loaded.' % (secs // 3600, (secs // 60) % 60, secs % 60))

    # Annealed learning rate.
    decay_steps = args.decay_steps if args.decay_steps is not None else args.epoch_steps
    lr = tf.keras.optimizers.schedules.ExponentialDecay(
        args.learning_rate, decay_steps, args.decay_rate
    )

    start_step = args.start_epoch * args.epoch_steps
    opt = tf.keras.optimizers.Adam(lr)
    opt.build(encoder.trainable_weights + decoder.trainable_weights + codebook.trainable_weights)
    opt.iterations.assign(start_step)

    # Setup codebook restarter
    restarter = CodebookRestarter(
        codebook.codebook_layer,
        32,
        random_init=(args.start_epoch == 0)
    )

    os.makedirs(args.output_dir, exist_ok=True)

    commit_weight_t = tf.constant(args.commit_weight, dtype=tf.float32)

    # Adam moment warmup: run N steps so the optimizer moment estimates (m, v)
    # settle, restoring all weights after each step so they stay frozen. Real
    # training then starts from a good gradient-variance estimate rather than
    # Adam's high-variance cold start. (m, v and the step count carry over.)
    if args.warmup_steps > 0:
        print(f"Warming up Adam: {args.warmup_steps} steps (moments accumulate, weights frozen)...")
        all_weights = (encoder.trainable_weights + decoder.trainable_weights +
                       codebook.trainable_weights)
        snapshot = [tf.identity(w) for w in all_weights]
        for _ in range(args.warmup_steps):
            train_step(
                encoder, decoder, codebook,
                opt, restarter, loader.random_batch(), loss_fn, commit_weight_t
            )
            for w, s in zip(all_weights, snapshot):
                w.assign(s)

    # Training loop
    epoch = args.start_epoch
    model_prefix = args.model
    while True:
        start_time = time.time()
        loss_acc = AverageAccumulator()
        audio_loss_acc = AverageAccumulator()
        commit_loss_acc = AverageAccumulator()
        nreset = 0

        for step in range(args.epoch_steps):
            batch = loader.random_batch()
            result = train_step(
                encoder, decoder, codebook,
                opt, restarter, batch, loss_fn, commit_weight_t
            )

            loss_acc.add(result['loss'])
            audio_loss_acc.add(result['audio_loss'])
            commit_loss_acc.add(result['commit_loss'])
            nreset += np.sum(result['reset'])

            etime = int(args.epoch_steps * ((time.time() - start_time) / (step + 1)))
            etime = '%02d:%02d:%02d' % (etime // 3600, (etime // 60) % 60, etime % 60)
            lr_value = opt.learning_rate
            if callable(lr_value):
                current_lr = float(lr_value(opt.iterations))
            else:
                current_lr = float(lr_value)

            print('Epoch=%04d Step=%04d Time=%s LR=%+.3e Loss=%+.3e (Audio=%+.3e Commit=%+.3e) Used=%05d Reset=%07d  ' %
                  (epoch, step, etime, current_lr, loss_acc.get(),
                   audio_loss_acc.get(), commit_loss_acc.get(),
                   np.sum(result['used']), nreset), end='\r')
        print()

        # Save latest weights every epoch.
        print(f"Saving weights (Audio={audio_loss_acc.get():.4e})")
        encoder.save_weights(os.path.join(args.output_dir, f'{model_prefix}_encoder.weights.h5'))
        decoder.save_weights(os.path.join(args.output_dir, f'{model_prefix}_decoder.weights.h5'))
        codebook.save_weights(os.path.join(args.output_dir, f'{model_prefix}_codebook.weights.h5'))

        epoch += 1


if __name__ == '__main__':
    main()
