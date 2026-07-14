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
from vqwave.util import AverageAccumulator


# GPU setup
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


@tf.function
def train_step(encoder, codebook, generator, optimizer, audio_batch, code_margin=0,
               label_smoothing=0.0):
    """
    Single training step for the generator.

    Args:
        encoder: VQ-VAE encoder (frozen, produces the codes we predict)
        codebook: VQ-VAE codebook (frozen)
        generator: Generator model
        optimizer: Optimizer
        audio_batch: Audio batch [batch, samples]
        code_margin: Number of codes to drop from each end before training. The
            'same'-padded encoder computes edge codes from zero-padding, so they
            are out-of-distribution; trimming them keeps the generator on codes
            with a full real-audio receptive field.
        label_smoothing: Softens the one-hot target (spreads this mass uniformly
            over the codebook) to curb overconfidence, which otherwise sharpens
            the next-token distribution and degrades free-running samples. 0
            disables it (and uses the cheaper sparse cross-entropy).

    Returns:
        dict with 'loss' and 'accuracy'
    """
    # Encode audio to codes (outside tape, frozen model)
    z_e = encoder(audio_batch, training=False)
    _, target_codes = codebook(z_e, training=False)

    # Drop padding-contaminated boundary codes.
    if code_margin > 0:
        target_codes = target_codes[:, code_margin:-code_margin]

    with tf.GradientTape() as tape:
        # Predict next code: input is codes[:-1], target is codes[1:]
        input_codes = target_codes[:, :-1]
        target_codes_shifted = target_codes[:, 1:]

        logits = generator(input_codes, training=True)

        if label_smoothing > 0.0:
            # Label smoothing needs one-hot targets; sparse CE has no such option.
            soft_targets = tf.one_hot(target_codes_shifted, tf.shape(logits)[-1])
            loss = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(
                    soft_targets, logits, from_logits=True,
                    label_smoothing=label_smoothing
                )
            )
        else:
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
                       help='Adam moment warmup: run N steps to settle the optimizer '
                            'moments (m, v) with weights frozen before training (default: 0, no warmup)')
    parser.add_argument('--input-length', type=int, default=None,
                       help='Input audio length in samples (transformers default to '
                            '(max_seq_len + 1 + 2*code_margin) * compression; others '
                            'default to 65536 plus the margin)')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                       help='Label smoothing for the next-code cross-entropy '
                            '(default: 0.1; 0 disables). Curbs overconfidence that '
                            'degrades free-running generation.')
    parser.add_argument('--code-margin', type=int, default=32,
                       help='Codes dropped from each end after encoding to discard '
                            'the zero-padded boundary codes the same-padded encoder '
                            'produces (default: 32, ~half the encoder receptive '
                            'field; 0 disables)')
    parser.add_argument('--epoch-steps', '--steps', type=int, default=10000,
                       help='Number of training steps per epoch (default: 10000)')
    parser.add_argument('--learning-rate', '--lr', type=float, default=1e-3,
                       help='Initial learning rate (default: 1e-3)')
    parser.add_argument('--decay-rate', type=float, default=0.9,
                       help='Multiplicative LR decay applied every --decay-steps (default: 0.9)')
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

    # For transformers the code-sequence length (input_length / compression) is
    # pinned to the position-embedding window. The next-token shift consumes one
    # code (input = codes[:-1], target = codes[1:]), so to supervise all
    # max_seq_len positions we need max_seq_len + 1 codes AFTER trimming
    # code_margin off each end, i.e. load max_seq_len + 1 + 2*code_margin codes.
    # Default it, and reject any mismatching override.
    compression = vqvae_config["compression_rate"]
    margin_samples = 2 * args.code_margin * compression
    if gen_config.get("type") == "transformer":
        max_seq_len = gen_config["transformer"].get("max_seq_len", 512)
        required_length = (max_seq_len + 1) * compression + margin_samples
        if args.input_length is None:
            args.input_length = required_length
            print(f"--input-length not set; defaulting to {required_length} "
                  f"((max_seq_len {max_seq_len} + 1 + 2*margin {args.code_margin}) "
                  f"* {compression}x compression)")
        elif args.input_length != required_length:
            seq_len = args.input_length // compression
            raise ValueError(
                f"--input-length {args.input_length} yields {seq_len} codes at "
                f"{compression}x compression, but '{args.generator}' has "
                f"max_seq_len={max_seq_len} and code_margin={args.code_margin}. Use "
                f"--input-length {required_length} (= (max_seq_len + 1 + 2*margin) "
                f"* compression) so the shift supervises all {max_seq_len} positions "
                f"after trimming."
            )
    elif args.input_length is None:
        # Add the trimmed margin so the retained sequence stays ~65536 samples.
        args.input_length = 2**16 + margin_samples
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
    opt = tf.keras.optimizers.Adam(base_lr, clipnorm=1.0)
    opt.build(generator.trainable_weights)
    opt.iterations.assign(start_step)

    # Adam moment warmup: run N steps so the optimizer moment estimates (m, v)
    # settle, restoring the weights after each step so they stay frozen. Real
    # training then starts from a good gradient-variance estimate rather than
    # Adam's high-variance cold start. (m, v and the step count carry over.)
    if args.warmup_steps > 0:
        print(f"Warming up Adam: {args.warmup_steps} steps (moments accumulate, weights frozen)...")
        snapshot = [tf.identity(w) for w in generator.trainable_weights]
        for _ in range(args.warmup_steps):
            train_step(encoder, codebook, generator, opt, loader.random_batch(),
                       args.code_margin, args.label_smoothing)
            for w, s in zip(generator.trainable_weights, snapshot):
                w.assign(s)

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
            result = train_step(encoder, codebook, generator, opt, batch,
                                args.code_margin, args.label_smoothing)

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
