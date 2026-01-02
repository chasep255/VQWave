#!/usr/bin/env python3
"""
Train generator model for autoregressive code prediction.

This script trains a generator to predict the next code in a sequence,
optionally conditioned on lower-resolution codes from a source VQ-VAE.
"""

import argparse
import os
import resource
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision

from vqwave.encoder import Encoder, CodebookManager
from vqwave.generator import create_generator
from vqwave.config import ENCODER_CONFIGS, GENERATOR_CONFIGS, SAMPLE_RATE
from vqwave.audio import AudioDataset
from vqwave.util import AverageAccumulator, LRWarmupWrapper


# GPU setup
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Increase file descriptor limit
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))


def train_step(dest_encoder, dest_codebook, generator, context_model, 
               source_encoder, source_codebook, optimizer, fp16, audio_batch):
    """
    Single training step for generator.
    
    Args:
        dest_encoder: Target VQ-VAE encoder (for codes we're predicting)
        dest_codebook: Target VQ-VAE codebook
        generator: Generator model
        context_model: Context model (None if unconditional)
        source_encoder: Source VQ-VAE encoder (None if unconditional)
        source_codebook: Source VQ-VAE codebook (None if unconditional)
        optimizer: Optimizer
        fp16: Whether using mixed precision
        audio_batch: Audio batch [batch, samples]
    
    Returns:
        dict with 'loss' and 'accuracy'
    """
    # Encode audio to target codes (outside tape, frozen model)
    z_e = dest_encoder(audio_batch, training=False)
    z_q, target_codes = dest_codebook(z_e, training=False)
    
    # Prepare context if needed (outside tape, frozen models)
    source_codes = None
    if context_model is not None:
        # Encode same audio to lower-res codes for context
        source_z_e = source_encoder(audio_batch, training=False)
        _, source_codes = source_codebook(source_z_e, training=False)
    
    # Training step inside gradient tape
    with tf.GradientTape() as tape:
        # Process through context model if needed
        context = None
        if context_model is not None:
            context = context_model(source_codes, training=True)
        
        # Generator predicts next code: input is codes[:-1], target is codes[1:]
        input_codes = target_codes[:, :-1]  # [batch, seq_len-1]
        target_codes_shifted = target_codes[:, 1:]  # [batch, seq_len-1]
        
        if context is not None:
            # Context should match sequence length, slice to match input_codes
            context = context[:, :tf.shape(input_codes)[1], :]  # [batch, seq_len-1, context_dim]
            logits = generator([input_codes, context], training=True)
        else:
            logits = generator(input_codes, training=True)
        
        # Sparse categorical crossentropy loss
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                target_codes_shifted, logits, from_logits=True
            )
        )
        
        # Compute accuracy
        predictions = tf.argmax(logits, axis=-1)
        accuracy = tf.reduce_mean(
            tf.cast(predictions == target_codes_shifted, tf.float32)
        )
    
    # Compute gradients and update
    weights = generator.trainable_weights
    if context_model is not None:
        weights = weights + context_model.trainable_weights
    
    grads = tape.gradient(loss, weights)
    optimizer.apply_gradients(zip(grads, weights))
    
    return {
        'loss': loss,
        'accuracy': accuracy,
    }


def main():
    parser = argparse.ArgumentParser(description='Train generator model for autoregressive code prediction')
    parser.add_argument('--generator', type=str, required=True,
                       choices=['generator_512', 'generator_128', 'generator_32', 'generator_8'],
                       help='Generator config name')
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
    parser.add_argument('--warmup-steps', type=int, default=0,
                       help='Number of warmup steps for learning rate (default: 0, no warmup)')
    parser.add_argument('--input-length', type=int, default=2**16,
                       help='Input audio length in samples (default: 65536)')
    parser.add_argument('--epoch-steps', '--steps', type=int, default=10000,
                       help='Number of training steps per epoch (default: 10000)')
    parser.add_argument('--learning-rate', '--lr', type=float, default=1e-3,
                       help='Initial learning rate (default: 1e-3)')
    parser.add_argument('--decay-rate', '--half-life', type=float, default=0.5,
                       help='Learning rate decay rate (default: 0.5, halves every decay_steps)')
    parser.add_argument('--decay-steps', type=int, default=None,
                       help='Number of steps for each decay (default: steps * 10)')
    parser.add_argument('--fp16', action='store_true', default=False,
                       help='Use mixed precision training (default: False)')
    
    args = parser.parse_args()
    
    # Set mixed precision policy
    if args.fp16:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Get generator config
    if args.generator not in GENERATOR_CONFIGS:
        available = ", ".join(GENERATOR_CONFIGS.keys())
        raise ValueError(f"Unknown generator preset '{args.generator}'. Available: {available}")
    
    gen_config = GENERATOR_CONFIGS[args.generator]
    dest_vqvae_key = gen_config["dest_vqvae"]
    source_vqvae_key = gen_config.get("source_vqvae")
    
    # Create target VQ-VAE (for codes we're predicting)
    dest_vqvae_config = ENCODER_CONFIGS[dest_vqvae_key]
    dest_encoder = Encoder(dest_vqvae_config)
    dest_codebook = CodebookManager(dest_vqvae_config)
    
    # Load target VQ-VAE weights (frozen during training)
    dest_encoder.load_weights(
        os.path.join(args.vqvae_weights_dir, f'{dest_vqvae_key}_encoder.weights.h5')
    )
    dest_codebook.load_weights(
        os.path.join(args.vqvae_weights_dir, f'{dest_vqvae_key}_codebook.weights.h5')
    )
    dest_encoder.trainable = False
    dest_codebook.trainable = False
    
    print(f"Loaded target VQ-VAE: {dest_vqvae_key}")
    print("Target Encoder:")
    dest_encoder.summary()
    print("\nTarget Codebook:")
    dest_codebook.summary()
    
    # Create source VQ-VAE and context model if needed
    source_encoder = None
    source_codebook = None
    context_model = None
    
    if source_vqvae_key is not None:
        source_vqvae_config = ENCODER_CONFIGS[source_vqvae_key]
        source_encoder = Encoder(source_vqvae_config)
        source_codebook = CodebookManager(source_vqvae_config)
        
        # Load source VQ-VAE weights (frozen during training)
        source_encoder.load_weights(
            os.path.join(args.vqvae_weights_dir, f'{source_vqvae_key}_encoder.weights.h5')
        )
        source_codebook.load_weights(
            os.path.join(args.vqvae_weights_dir, f'{source_vqvae_key}_codebook.weights.h5')
        )
        source_encoder.trainable = False
        source_codebook.trainable = False
        
        print(f"\nLoaded source VQ-VAE: {source_vqvae_key}")
        print("Source Encoder:")
        source_encoder.summary()
        print("\nSource Codebook:")
        source_codebook.summary()
    
    # Create generator and context model
    generator, context_model = create_generator(args.generator)
    
    print("\nGenerator:")
    generator.summary()
    if context_model is not None:
        print("\nContext Model:")
        context_model.summary()
    
    # Load generator/context weights if resuming
    if args.start_epoch > 0:
        generator.load_weights(
            os.path.join(args.output_dir, f'{args.generator}_generator.weights.h5')
        )
        if context_model is not None:
            context_model.load_weights(
                os.path.join(args.output_dir, f'{args.generator}_context.weights.h5')
            )
        print(f"Resuming training from epoch {args.start_epoch}")
    
    # Load dataset
    data = AudioDataset(args.data_dir)
    secs = data.total_samples / SAMPLE_RATE
    print(f'\n%02d:%02d:%02d of training audio loaded.' % (secs // 3600, (secs // 60) % 60, secs % 60))
    
    # Setup optimizer with learning rate schedule
    decay_steps = args.decay_steps if args.decay_steps is not None else args.epoch_steps * 10
    base_lr = tf.keras.optimizers.schedules.ExponentialDecay(
        args.learning_rate, decay_steps, args.decay_rate
    )
    
    # Add warmup if requested
    start_step = args.start_epoch * args.epoch_steps
    if args.warmup_steps > 0:
        # growth_rate = 1 / warmup_steps to reach full LR after warmup_steps
        growth_rate = 1.0 / args.warmup_steps
        lr = LRWarmupWrapper(base_lr, growth_rate=growth_rate, initial_step=start_step)
        print(f"Using LR warmup for first {args.warmup_steps} steps from LR=0 (starting from step {start_step})")
    else:
        lr = base_lr
    
    opt = tf.keras.optimizers.Adam(lr, clipnorm=1.0)
    opt.iterations.assign(start_step)
    if args.fp16:
        opt = tf.keras.mixed_precision.LossScaleOptimizer(opt, dynamic_growth_steps=512)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    epoch = args.start_epoch
    best_loss = float('inf')
    while True:
        start_time = time.time()
        loss_acc = AverageAccumulator()
        accuracy_acc = AverageAccumulator()
        
        for step in range(args.epoch_steps):
            batch = data.random_batch(args.batch_size, args.input_length)[0]
            result = train_step(
                dest_encoder, dest_codebook, generator, context_model,
                source_encoder, source_codebook, opt, args.fp16, batch
            )
            
            loss_acc.add(result['loss'])
            accuracy_acc.add(result['accuracy'])
            
            etime = int(args.epoch_steps * ((time.time() - start_time) / (step + 1)))
            etime = '%02d:%02d:%02d' % (etime // 3600, (etime // 60) % 60, etime % 60)
            # Get learning rate - handle both schedule and wrapped optimizer
            if hasattr(opt, 'inner_optimizer'):
                lr_value = opt.inner_optimizer.learning_rate
            else:
                lr_value = opt.learning_rate
            if callable(lr_value):
                current_lr = float(lr_value(opt.iterations))
            else:
                current_lr = float(lr_value)
            print('Epoch=%04d Step=%04d Time=%s LR=%+.4e Loss=%+.4e Acc=%0.4f  ' % 
                  (epoch, step, etime, current_lr, loss_acc.get(), accuracy_acc.get()), end='\r')
        print()
        
        # Get average loss for this epoch
        current_loss = loss_acc.get()
        
        # Save weights only if loss improved
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
            if context_model is not None:
                context_model.save_weights(
                    os.path.join(args.output_dir, f'{args.generator}_context.weights.h5')
                )
        else:
            print(f"Skipping save (loss did not improve: {current_loss:.4e} >= {best_loss:.4e})")
        
        epoch += 1


if __name__ == '__main__':
    main()

