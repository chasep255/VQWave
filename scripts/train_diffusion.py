#!/usr/bin/env python3
"""
Train the diffusion decoder that renders a waveform from the VQ codes.

The VQ-VAE encoder + codebook are frozen and used only to produce the quantized
code embeddings z_q that condition the diffusion model. The denoiser is trained
with v-prediction (MSE against the v-target) on a continuous-time cosine schedule.
"""

import argparse
import os
import time

import tensorflow as tf
from tensorflow import keras

from vqwave.encoder import Encoder, Decoder, CodebookManager
from vqwave.diffusion import Denoiser, normalize
from vqwave.config import DIFFUSION_CONFIGS, ENCODER_CONFIGS, SAMPLE_RATE
from vqwave.audio import AudioLoader
from vqwave.util import AverageAccumulator


# GPU setup
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def main():
    parser = argparse.ArgumentParser(description='Train the diffusion decoder (codes -> waveform)')
    parser.add_argument('--diffusion', type=str, required=True,
                       choices=list(DIFFUSION_CONFIGS.keys()),
                       help=f'Diffusion config name (choices: {", ".join(DIFFUSION_CONFIGS.keys())})')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing training audio files')
    parser.add_argument('--vqvae-weights-dir', type=str, default='weights',
                       help='Directory with VQ-VAE weights (default: weights)')
    parser.add_argument('--output-dir', type=str, default='weights',
                       help='Directory to save diffusion weights (default: weights)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size (default: 8)')
    parser.add_argument('--start-epoch', type=int, default=0,
                       help='Starting epoch number (default: 0)')
    parser.add_argument('--load-weights', action='store_true', default=False,
                       help='Load existing diffusion weights from output directory (default: False)')
    parser.add_argument('--input-length', type=int, default=2**16,
                       help='Input audio length in samples (must be a multiple of the '
                            'compression rate; default: 65536)')
    parser.add_argument('--epoch-steps', '--steps', type=int, default=10000,
                       help='Number of training steps per epoch (default: 10000)')
    parser.add_argument('--learning-rate', '--lr', type=float, default=1e-4,
                       help='Initial learning rate (default: 1e-4)')
    parser.add_argument('--decay-rate', type=float, default=0.9,
                       help='Multiplicative LR decay applied every --decay-steps (default: 0.9)')
    parser.add_argument('--decay-steps', type=int, default=None,
                       help='Number of steps for each decay (default: epoch_steps)')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                       help='Global grad-norm clip (0 disables; default: 1.0)')
    parser.add_argument('--warmup-steps', type=int, default=0,
                       help='Adam moment warmup: run N steps to settle the optimizer '
                            'moments (m, v) with weights frozen before training (default: 0, no warmup)')
    parser.add_argument('--bf16', action='store_true', default=False,
                       help='Use mixed_bfloat16 precision (Ampere+; half memory, no loss scaling)')

    args = parser.parse_args()

    if args.bf16:
        keras.mixed_precision.set_global_policy('mixed_bfloat16')
        print("Using mixed_bfloat16 precision")

    config = DIFFUSION_CONFIGS[args.diffusion]
    dest_vqvae = config["dest_vqvae"]
    vqvae_config = ENCODER_CONFIGS[dest_vqvae]
    compression = vqvae_config["compression_rate"]

    if args.input_length % compression != 0:
        raise ValueError(
            f"--input-length ({args.input_length}) must be a multiple of the "
            f"compression rate ({compression})")

    # Frozen VQ-VAE (encoder + codebook + decoder) produces the deterministic
    # reconstruction the diffusion model is conditioned on.
    encoder = Encoder(vqvae_config)
    codebook = CodebookManager(vqvae_config)
    decoder = Decoder(vqvae_config)
    encoder.load_weights(os.path.join(args.vqvae_weights_dir, f'{dest_vqvae}_encoder.weights.h5'))
    codebook.load_weights(os.path.join(args.vqvae_weights_dir, f'{dest_vqvae}_codebook.weights.h5'))
    decoder.load_weights(os.path.join(args.vqvae_weights_dir, f'{dest_vqvae}_decoder.weights.h5'))
    encoder.trainable = False
    codebook.trainable = False
    decoder.trainable = False
    print(f"Loaded frozen VQ-VAE: {dest_vqvae}")

    denoiser = Denoiser(config)
    print("\nDiffusion decoder:")
    denoiser.summary()

    # Load weights if resuming training or --load-weights flag is set
    if args.start_epoch > 0 or args.load_weights:
        denoiser_path = os.path.join(args.output_dir, f'{args.diffusion}_denoiser.weights.h5')
        if os.path.exists(denoiser_path):
            print(f"\nLoading diffusion weights from {denoiser_path}...")
            denoiser.load_weights(denoiser_path)
            print("Weights loaded successfully.")
        else:
            raise FileNotFoundError(f"Diffusion weights not found. Expected: {denoiser_path}")

    # Load dataset (background-threaded prefetcher; reads crops on demand)
    loader = AudioLoader(args.data_dir, args.input_length, args.batch_size)
    secs = loader.total_samples / SAMPLE_RATE
    print('\n%02d:%02d:%02d of training audio loaded.' % (secs // 3600, (secs // 60) % 60, secs % 60))

    decay_steps = args.decay_steps if args.decay_steps is not None else args.epoch_steps
    lr = tf.keras.optimizers.schedules.ExponentialDecay(
        args.learning_rate, decay_steps, args.decay_rate
    )
    start_step = args.start_epoch * args.epoch_steps
    opt = tf.keras.optimizers.Adam(lr, global_clipnorm=args.grad_clip or None)
    opt.build(denoiser.trainable_weights)
    opt.iterations.assign(start_step)

    os.makedirs(args.output_dir, exist_ok=True)

    @tf.function
    def train_step(audio_batch):
        # Deterministic VQ-VAE reconstruction -> the conditioning waveform.
        z_e = encoder(audio_batch, training=False)
        z_q, _ = codebook(z_e, training=False)
        decoded = decoder(z_q, training=False)
        cond = normalize(decoded)

        # Diffusion target lives in the normalized (companded) domain.
        x = normalize(audio_batch)
        b = tf.shape(x)[0]
        # Low-discrepancy timesteps: stratified across [0, 1] with per-sample jitter.
        t = (tf.cast(tf.range(b), tf.float32) + tf.random.uniform((b,))) / tf.cast(b, tf.float32)
        noise = tf.random.normal(tf.shape(x))
        x_t = denoiser.merge_noise(x, noise, t)
        target = denoiser.target(x, noise, t)

        with tf.GradientTape() as tape:
            pred = denoiser((x_t, t, cond), training=True)
            loss = tf.reduce_mean(tf.square(pred - target))

        grads = tape.gradient(loss, denoiser.trainable_weights)
        opt.apply_gradients(zip(grads, denoiser.trainable_weights))
        return loss

    # Adam moment warmup: run N steps so the optimizer moment estimates (m, v)
    # settle, restoring the weights after each step so they stay frozen. Real
    # training then starts from a good gradient-variance estimate rather than
    # Adam's high-variance cold start. (m, v and the step count carry over.)
    if args.warmup_steps > 0:
        print(f"Warming up Adam: {args.warmup_steps} steps (moments accumulate, weights frozen)...")
        snapshot = [tf.identity(w) for w in denoiser.trainable_weights]
        for _ in range(args.warmup_steps):
            train_step(loader.random_batch())
            for w, s in zip(denoiser.trainable_weights, snapshot):
                w.assign(s)

    # Training loop
    epoch = args.start_epoch
    while True:
        start_time = time.time()
        loss_acc = AverageAccumulator()

        for step in range(args.epoch_steps):
            loss_acc.add(float(train_step(loader.random_batch())))

            etime = int(args.epoch_steps * ((time.time() - start_time) / (step + 1)))
            etime = '%02d:%02d:%02d' % (etime // 3600, (etime // 60) % 60, etime % 60)
            current_lr = float(lr(opt.iterations))
            print('Epoch=%04d Step=%04d Time=%s LR=%+.3e Loss=%+.4e  ' %
                  (epoch, step, etime, current_lr, loss_acc.get()), end='\r')
        print()

        print(f"Saving weights (Loss={loss_acc.get():.4e})")
        denoiser.save_weights(os.path.join(args.output_dir, f'{args.diffusion}_denoiser.weights.h5'))
        epoch += 1


if __name__ == '__main__':
    main()
