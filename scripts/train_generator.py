#!/usr/bin/env python3
"""
Train the generator for autoregressive code prediction.

The generator predicts the next 256x VQ-VAE code from the previous codes
(unconditional). Either generator architecture (transformer or rnn) may be used.
"""

import argparse
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

from vqwave.encoder import Encoder, CodebookManager
from vqwave.generator import create_generator
from vqwave.config import ENCODER_CONFIGS, GENERATOR_CONFIGS, SAMPLE_RATE
from vqwave.audio import AudioLoader
from vqwave.util import AverageAccumulator, LRWarmupWrapper


# GPU setup
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


@tf.function
def train_step(encoder, codebook, generator, optimizer, audio_batch):
    """
    Single training step for the generator.

    Args:
        encoder: VQ-VAE encoder (frozen, produces the codes we predict)
        codebook: VQ-VAE codebook (frozen)
        generator: Generator model
        optimizer: Optimizer
        audio_batch: Audio batch [batch, samples]

    Returns:
        dict with 'loss' and 'accuracy'
    """
    # Encode audio to codes (outside tape, frozen model)
    z_e = encoder(audio_batch, training=False)
    _, target_codes = codebook(z_e, training=False)

    with tf.GradientTape() as tape:
        # Predict next code: input is codes[:-1], target is codes[1:]
        input_codes = target_codes[:, :-1]
        target_codes_shifted = target_codes[:, 1:]

        logits = generator(input_codes, training=True)

        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                target_codes_shifted, logits, from_logits=True
            )
        )

        predictions = tf.argmax(logits, axis=-1)
        accuracy = tf.reduce_mean(
            tf.cast(predictions == target_codes_shifted, tf.float32)
        )

    grads = tape.gradient(loss, generator.trainable_weights)
    optimizer.apply_gradients(zip(grads, generator.trainable_weights))

    return {
        'loss': loss,
        'accuracy': accuracy,
    }


def main():
    parser = argparse.ArgumentParser(description='Train generator for autoregressive code prediction')
    parser.add_argument('--generator', type=str, required=True,
                       choices=list(GENERATOR_CONFIGS.keys()),
                       help=f'Generator config name (choices: {", ".join(GENERATOR_CONFIGS.keys())})')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing training audio files')
    parser.add_argument('--vqvae-weights-dir', type=str, default='weights',
                       help='Directory with VQ-VAE weights (default: weights)')
    parser.add_argument('--output-dir', type=str, default='weights',
                       help='Directory to save generator weights (default: weights)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size (default: 8)')
    parser.add_argument('--start-epoch', type=int, default=0,
                       help='Starting epoch number (default: 0)')
    parser.add_argument('--load-weights', action='store_true', default=False,
                       help='Load existing weights from output directory (default: False)')
    parser.add_argument('--warmup-steps', type=int, default=0,
                       help='Number of warmup steps for learning rate (default: 0, no warmup)')
    parser.add_argument('--input-length', type=int, default=2**16,
                       help='Input audio length in samples (default: 65536)')
    parser.add_argument('--epoch-steps', '--steps', type=int, default=10000,
                       help='Number of training steps per epoch (default: 10000)')
    parser.add_argument('--learning-rate', '--lr', type=float, default=1e-3,
                       help='Initial learning rate (default: 1e-3)')
    parser.add_argument('--decay-rate', type=float, default=0.85,
                       help='Multiplicative LR decay applied every --decay-steps (default: 0.85)')
    parser.add_argument('--decay-steps', type=int, default=None,
                       help='Number of steps for each decay (default: epoch_steps)')
    parser.add_argument('--bf16', action='store_true', default=False,
                       help='Use mixed_bfloat16 precision (Ampere+; half memory, no loss scaling)')

    args = parser.parse_args()

    # Mixed precision must be set before models are built.
    if args.bf16:
        keras.mixed_precision.set_global_policy('mixed_bfloat16')
        print("Using mixed_bfloat16 precision")

    gen_config = GENERATOR_CONFIGS[args.generator]
    dest_vqvae_key = gen_config["dest_vqvae"]

    # Create VQ-VAE (for codes we're predicting), frozen during training
    vqvae_config = ENCODER_CONFIGS[dest_vqvae_key]
    encoder = Encoder(vqvae_config)
    codebook = CodebookManager(vqvae_config)

    encoder.load_weights(
        os.path.join(args.vqvae_weights_dir, f'{dest_vqvae_key}_encoder.weights.h5')
    )
    codebook.load_weights(
        os.path.join(args.vqvae_weights_dir, f'{dest_vqvae_key}_codebook.weights.h5')
    )
    encoder.trainable = False
    codebook.trainable = False

    print(f"Loaded VQ-VAE: {dest_vqvae_key}")
    print("Encoder:")
    encoder.summary()
    print("\nCodebook:")
    codebook.summary()

    # Create generator
    generator = create_generator(args.generator)
    print("\nGenerator:")
    generator.summary()

    # Load weights if resuming training or --load-weights flag is set
    if args.start_epoch > 0 or args.load_weights:
        generator_path = os.path.join(args.output_dir, f'{args.generator}_generator.weights.h5')
        if os.path.exists(generator_path):
            print(f"\nLoading weights from {args.output_dir}...")
            generator.load_weights(generator_path)
            print("Weights loaded successfully.")
            if args.start_epoch > 0:
                print(f"Resuming training from epoch {args.start_epoch}")
        else:
            raise FileNotFoundError(
                f"Generator weights not found. Expected: {generator_path}"
            )

    # Load dataset (background-threaded prefetcher; reads crops on demand)
    loader = AudioLoader(args.data_dir, args.input_length, args.batch_size)
    secs = loader.total_samples / SAMPLE_RATE
    print('\n%02d:%02d:%02d of training audio loaded.' % (secs // 3600, (secs // 60) % 60, secs % 60))

    # Setup optimizer with learning rate schedule
    decay_steps = args.decay_steps if args.decay_steps is not None else args.epoch_steps
    base_lr = tf.keras.optimizers.schedules.ExponentialDecay(
        args.learning_rate, decay_steps, args.decay_rate
    )

    start_step = args.start_epoch * args.epoch_steps
    if args.warmup_steps > 0:
        growth_rate = 1.0 / args.warmup_steps
        lr = LRWarmupWrapper(base_lr, growth_rate=growth_rate, initial_step=start_step)
        print(f"Using LR warmup for first {args.warmup_steps} steps from LR=0 (starting from step {start_step})")
    else:
        lr = base_lr

    opt = tf.keras.optimizers.Adam(lr, clipnorm=1.0)
    opt.iterations.assign(start_step)

    os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    epoch = args.start_epoch
    best_loss = float('inf')
    while True:
        start_time = time.time()
        loss_acc = AverageAccumulator()
        accuracy_acc = AverageAccumulator()

        for step in range(args.epoch_steps):
            batch = loader.random_batch()
            result = train_step(encoder, codebook, generator, opt, batch)

            loss_acc.add(result['loss'])
            accuracy_acc.add(result['accuracy'])

            etime = int(args.epoch_steps * ((time.time() - start_time) / (step + 1)))
            etime = '%02d:%02d:%02d' % (etime // 3600, (etime // 60) % 60, etime % 60)
            lr_value = opt.learning_rate
            if callable(lr_value):
                current_lr = float(lr_value(opt.iterations))
            else:
                current_lr = float(lr_value)
            print('Epoch=%04d Step=%04d Time=%s LR=%+.4e Loss=%+.4e Acc=%0.4f  ' %
                  (epoch, step, etime, current_lr, loss_acc.get(), accuracy_acc.get()), end='\r')
        print()

        current_loss = loss_acc.get()
        if current_loss < best_loss:
            prev_best = best_loss
            best_loss = current_loss
            if prev_best == float('inf'):
                print(f"Saving weights (initial save, loss: {current_loss:.4e})")
            else:
                print(f"Saving weights (loss improved: {current_loss:.4e} < {prev_best:.4e})")
            generator.save_weights(
                os.path.join(args.output_dir, f'{args.generator}_generator.weights.h5')
            )
        else:
            print(f"Skipping save (loss did not improve: {current_loss:.4e} >= {best_loss:.4e})")

        epoch += 1


if __name__ == '__main__':
    main()
