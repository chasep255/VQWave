#!/usr/bin/env python3
"""
Train VQ-VAE encoder/decoder model.

This script trains the encoder, decoder, and codebook components of a VQ-VAE model
using audio reconstruction loss.
"""

import argparse
import os
import resource
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision

from vqwave.encoder import Encoder, Decoder, CodebookManager
from vqwave.config import ENCODER_CONFIGS, SAMPLE_RATE
from vqwave.audio import AudioDataset
from vqwave.util import AverageAccumulator, CodebookRestarter, LRWarmupWrapper


# GPU setup
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Increase file descriptor limit
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))


@tf.function
def stft_loss(y, r):
    """Compute multi-scale STFT loss."""
    loss = 0.0
    for w, s in ((256, 64), (512, 128), (1024, 256), (2048, 512)):
        y_ = tf.signal.stft(y, w, s)
        r_ = tf.signal.stft(r, w, s)
        loss += tf.reduce_mean(tf.abs(y_ - r_))

    return loss



def train_step(encoder, decoder, codebook, optimizer, restarter, fp16, r):
    """Single training step."""
    with tf.GradientTape() as tape:
        # Encode audio to latents
        z_e = encoder(r, training=True)
        
        # Quantize latents
        z_q, codes = codebook(z_e, training=True)
        
        # Straight-through estimator: forward uses z_q, backward passes through z_e
        z_q_st = z_e + tf.stop_gradient(z_q - z_e)
        
        # Decode to audio
        y = decoder(z_q_st, training=True)
        
        audio_loss = stft_loss(y, r)

        # VQ-VAE commitment loss: updates both encoder and codebook
        commit_loss = tf.reduce_mean(tf.square(z_e - z_q))

        loss = audio_loss + commit_loss
    
    weights = (decoder.trainable_weights + 
               codebook.trainable_weights + 
               encoder.trainable_weights)
    grads = tape.gradient(loss, weights)
    # LossScaleOptimizer handles scaling automatically in newer TensorFlow
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
    parser = argparse.ArgumentParser(description='Train VQ-VAE encoder/decoder model')
    parser.add_argument('--model', type=str, required=True,
                       choices=['vqvae_512', 'vqvae_128', 'vqvae_32', 'vqvae_8'],
                       help='Model preset name')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing training audio files')
    parser.add_argument('--output-dir', type=str, default='weights',
                       help='Directory to save model weights (default: weights)')
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
    parser.add_argument('--code-reset-limit', type=int, default=32,
                       help='Codebook reset limit (default: 32)')
    parser.add_argument('--learning-rate', '--lr', type=float, default=2e-4,
                       help='Initial learning rate (default: 2e-4)')
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
    
    # Create encoder, decoder, and codebook
    if args.model not in ENCODER_CONFIGS:
        available = ", ".join(ENCODER_CONFIGS.keys())
        raise ValueError(f"Unknown encoder preset '{args.model}'. Available: {available}")
    
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
    
    # Load weights if resuming training
    if args.start_epoch > 0:
        model_prefix = args.model
        encoder.load_weights(
            os.path.join(args.output_dir, f'{model_prefix}_encoder.weights.h5')
        )
        decoder.load_weights(
            os.path.join(args.output_dir, f'{model_prefix}_decoder.weights.h5')
        )
        codebook.load_weights(
            os.path.join(args.output_dir, f'{model_prefix}_codebook.weights.h5')
        )
    
    # Load dataset
    data = AudioDataset(args.data_dir)
    secs = data.total_samples / SAMPLE_RATE
    print('%02d:%02d:%02d of training audio loaded.' % (secs // 3600, (secs // 60) % 60, secs % 60))
    
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
    
    opt = tf.keras.optimizers.Adam(lr)
    opt.iterations.assign(start_step)
    if args.fp16:
        opt = tf.keras.mixed_precision.LossScaleOptimizer(opt, dynamic_growth_steps=512)
    
    # Setup codebook restarter
    restarter = CodebookRestarter(
        codebook.codebook_layer,
        32,
        random_init=(args.start_epoch == 0)
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    epoch = args.start_epoch
    while True:
        start_time = time.time()
        loss_acc = AverageAccumulator()
        audio_loss_acc = AverageAccumulator()
        commit_loss_acc = AverageAccumulator()
        nreset = 0
        
        for step in range(args.epoch_steps):
            batch = data.random_batch(args.batch_size, args.input_length)[0]
            result = train_step(encoder, decoder, codebook, opt, restarter, args.fp16, batch)
            
            loss_acc.add(result['loss'])
            audio_loss_acc.add(result['audio_loss'])
            commit_loss_acc.add(result['commit_loss'])
            nreset += np.sum(result['reset'])
            
            etime = int(args.epoch_steps * ((time.time() - start_time) / (step + 1)))
            etime = '%02d:%02d:%02d' % (etime // 3600, (etime // 60) % 60, etime % 60)
            # Get learning rate - handle both schedule and wrapped optimizer
            if hasattr(opt, 'inner_optimizer'):
                # LossScaleOptimizer wraps the optimizer
                lr_value = opt.inner_optimizer.learning_rate
            else:
                lr_value = opt.learning_rate
            if callable(lr_value):
                current_lr = float(lr_value(opt.iterations))
            else:
                current_lr = float(lr_value)
            print('Epoch=%04d Step=%04d Time=%s LR=%+.4e Loss=%+.4e Audio-Loss=%+.4e Commit-Loss=%+.4e Used=%05d Reset=%07d  ' % 
                  (epoch, step, etime, current_lr, loss_acc.get(), audio_loss_acc.get(), 
                   commit_loss_acc.get(), np.sum(result['used']), nreset), end='\r')
        print()
        
        # Save weights without epoch numbers (overwrites each epoch)
        model_prefix = args.model
        encoder.save_weights(
            os.path.join(args.output_dir, f'{model_prefix}_encoder.weights.h5')
        )
        decoder.save_weights(
            os.path.join(args.output_dir, f'{model_prefix}_decoder.weights.h5')
        )
        codebook.save_weights(
            os.path.join(args.output_dir, f'{model_prefix}_codebook.weights.h5')
        )
        
        epoch += 1


if __name__ == '__main__':
    main()

