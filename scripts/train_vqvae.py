#!/usr/bin/env python3
"""
Train VQ-VAE encoder/decoder model.

This script trains the encoder, decoder, and codebook components of a VQ-VAE model
using spectrogram-based loss functions.
"""

import argparse
import os
import resource
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision

from lib.encoder import Encoder, Decoder, CodebookManager
from lib.config import ENCODER_CONFIGS, SAMPLE_RATE
from lib.audio import AudioDataset
from lib.util import AverageAccumulator, CodebookRestarter


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
    for w, s in ((220, 44), (770, 110), (1100, 240)):
        y_ = tf.signal.stft(y, w, s)
        r_ = tf.signal.stft(r, w, s)
        loss += tf.reduce_mean(tf.square(tf.abs(y_ - r_)))
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
        
        spectral_loss = stft_loss(y, r)

        # VQ-VAE losses:
        # - codebook loss updates codebook embeddings
        # - commitment loss updates encoder (beta term)
        z_e_f32 = tf.cast(z_e, tf.float32)
        z_q_f32 = tf.cast(z_q, tf.float32)
        codebook_loss = tf.reduce_mean(tf.square(tf.stop_gradient(z_e_f32) - z_q_f32))
        commit_loss = 0.1 * tf.reduce_mean(tf.square(z_e_f32 - tf.stop_gradient(z_q_f32)))

        loss = spectral_loss + codebook_loss + commit_loss
    
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
        'spectral_loss': spectral_loss,
        'codebook_loss': codebook_loss,
        'commit_loss': commit_loss,
        'used': num_used,
        'reset': num_reset,
    }


def main():
    parser = argparse.ArgumentParser(description='Train VQ-VAE encoder/decoder model')
    parser.add_argument('--model', type=str, required=True,
                       choices=['vqvae_512', 'vqvae_128', 'vqvae_32'],
                       help='Model preset name')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing training audio files')
    parser.add_argument('--output-dir', type=str, default='weights',
                       help='Directory to save model weights (default: weights)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size (default: 8)')
    parser.add_argument('--start-epoch', type=int, default=0,
                       help='Starting epoch number (default: 0)')
    parser.add_argument('--input-length', type=int, default=2**16,
                       help='Input audio length in samples (default: 65536)')
    parser.add_argument('--steps', type=int, default=10000,
                       help='Number of training steps per epoch (default: 10000)')
    parser.add_argument('--code-reset-limit', type=int, default=32,
                       help='Codebook reset limit (default: 32)')
    parser.add_argument('--fp16', action='store_true', default=True,
                       help='Use mixed precision training (default: True)')
    parser.add_argument('--no-fp16', dest='fp16', action='store_false',
                       help='Disable mixed precision training')
    
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
        prev_epoch = args.start_epoch - 1
        encoder.load_weights(
            os.path.join(args.output_dir, f'{model_prefix}_encoder_{prev_epoch:05d}.weights.h5')
        )
        decoder.load_weights(
            os.path.join(args.output_dir, f'{model_prefix}_decoder_{prev_epoch:05d}.weights.h5')
        )
        codebook.load_weights(
            os.path.join(args.output_dir, f'{model_prefix}_codebook_{prev_epoch:05d}.weights.h5')
        )
    
    # Load dataset
    data = AudioDataset(args.data_dir)
    secs = data.total_samples / SAMPLE_RATE
    print('%02d:%02d:%02d of training audio loaded.' % (secs // 3600, (secs // 60) % 60, secs % 60))
    
    # Setup optimizer with learning rate schedule
    lr = tf.keras.optimizers.schedules.ExponentialDecay(2e-4, args.steps * 10, 0.5)
    opt = tf.keras.optimizers.Adam(lr)
    opt.iterations.assign(args.start_epoch * args.steps)
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
        spectral_loss_acc = AverageAccumulator()
        codebook_loss_acc = AverageAccumulator()
        commit_loss_acc = AverageAccumulator()
        nreset = 0
        
        for step in range(args.steps):
            batch = data.random_batch(args.batch_size, args.input_length)[0]
            result = train_step(encoder, decoder, codebook, opt, restarter, args.fp16, batch)
            
            loss_acc.add(result['loss'])
            spectral_loss_acc.add(result['spectral_loss'])
            codebook_loss_acc.add(result['codebook_loss'])
            commit_loss_acc.add(result['commit_loss'])
            nreset += np.sum(result['reset'])
            
            etime = int(args.steps * ((time.time() - start_time) / (step + 1)))
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
            print('Epoch=%04d Step=%04d Time=%s LR=%+.4e Loss=%+.4e Spectral-Loss=%+.4e Codebook-Loss=%+.4e Commit-Loss=%+.4e Used=%05d Reset=%07d  ' % 
                  (epoch, step, etime, current_lr, loss_acc.get(), spectral_loss_acc.get(), 
                   codebook_loss_acc.get(), commit_loss_acc.get(), np.sum(result['used']), nreset), end='\r')
        print()
        
        # Save weights with model name prefix: {model_name}_{component}_{epoch}.weights.h5
        model_prefix = args.model
        encoder.save_weights(
            os.path.join(args.output_dir, f'{model_prefix}_encoder_{epoch:05d}.weights.h5')
        )
        decoder.save_weights(
            os.path.join(args.output_dir, f'{model_prefix}_decoder_{epoch:05d}.weights.h5')
        )
        codebook.save_weights(
            os.path.join(args.output_dir, f'{model_prefix}_codebook_{epoch:05d}.weights.h5')
        )
        
        epoch += 1


if __name__ == '__main__':
    main()

