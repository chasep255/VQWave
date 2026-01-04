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
    loss = []
    for w, s in ((220, 41), (770, 109), (1100, 239)):
        y_ = tf.abs(tf.signal.stft(y, w, s))
        r_ = tf.abs(tf.signal.stft(r, w, s))
        loss.append(tf.reduce_mean(tf.abs(y_ - r_)))
    return tf.reduce_mean(loss)


@tf.function
def perceptual_mel_loss(y, r, sample_rate=22050, n_fft=1024, hop_length=256, n_mels=80):
    """
    Mel spectrogram loss in linear (non-log) space.
    
    Args:
        y: Generated audio [batch, samples]
        r: Reference audio [batch, samples]
        sample_rate: Audio sample rate (default: 22050)
        n_fft: FFT window size (default: 1024, standard for 22kHz audio)
        hop_length: Hop length for STFT (default: 256, typically n_fft/4)
        n_mels: Number of mel filter banks (default: 80)
    """
    y_stft = tf.signal.stft(y, n_fft, hop_length, fft_length=n_fft)
    r_stft = tf.signal.stft(r, n_fft, hop_length, fft_length=n_fft)

    y_mag = tf.abs(y_stft)
    r_mag = tf.abs(r_stft)

    mel_w = tf.signal.linear_to_mel_weight_matrix(
        n_mels, n_fft // 2 + 1, sample_rate, 0.0, sample_rate / 2.0
    )

    y_mel = tf.einsum('btf,fm->btm', y_mag, mel_w)
    r_mel = tf.einsum('btf,fm->btm', r_mag, mel_w)

    # L1 loss on raw mel spectrograms (linear space)
    return tf.reduce_mean(tf.abs(y_mel - r_mel))


@tf.function
def hierarchical_l1_loss(y, r, scales=[256, 128, 64, 32, 16, 8, 4, 2, 1]):
    """
    Hierarchical L1 loss computed at multiple temporal scales.
    
    Computes L1 loss at different downsampling rates and averages them.
    This encourages the model to match the reference at multiple temporal resolutions.
    
    Args:
        y: Generated audio [batch, samples]
        r: Reference audio [batch, samples]
        scales: List of downsampling factors (default: [256, 128, 64, 32, 16, 8, 4, 2, 1])
    
    Returns:
        Average L1 loss across all scales
    """
    loss = 0.0
    
    for scale in scales:
        if scale == 1:
            # Original resolution - no downsampling
            loss += tf.reduce_mean(tf.abs(y - r))
        else:
            # Downsample by averaging over windows
            # Add channel dimension: [batch, samples] -> [batch, samples, 1]
            y_expanded = tf.expand_dims(y, axis=-1)  # [batch, samples, 1]
            r_expanded = tf.expand_dims(r, axis=-1)  # [batch, samples, 1]
            
            # Average pool with pool_size=scale, stride=scale
            y_down = tf.nn.avg_pool1d(y_expanded, ksize=scale, strides=scale, padding='VALID')
            r_down = tf.nn.avg_pool1d(r_expanded, ksize=scale, strides=scale, padding='VALID')
            
            # Remove channel dimension: [batch, samples/scale, 1] -> [batch, samples/scale]
            y_down = tf.squeeze(y_down, axis=-1)
            r_down = tf.squeeze(r_down, axis=-1)
            
            # Compute L1 loss at this scale
            loss += tf.reduce_mean(tf.abs(y_down - r_down))
        
    return loss


@tf.function
def mse_loss(y, r):
    """
    Compute simple mean squared error loss on waveforms.
    
    Args:
        y: Generated audio [batch, samples]
        r: Reference audio [batch, samples]
    """
    return tf.reduce_mean(tf.square(y - r))


# Loss function registry
LOSS_FUNCTIONS = {
    'stft': stft_loss,
    'mel': perceptual_mel_loss,
    'mse': mse_loss,
}


def train_step(encoder, decoder, codebook, optimizer, restarter, r):
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
        
        # STFT audio loss (multi-scale)
        stft_loss_val = stft_loss(y, r)
        audio_loss = stft_loss_val

        # VQ-VAE commitment loss: updates both encoder and codebook
        commit_loss = tf.reduce_mean(tf.square(z_e - z_q))

        loss = audio_loss + 0.01 * commit_loss
    
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
        'stft_loss': stft_loss_val,
        'commit_loss': commit_loss,
        'used': num_used,
        'reset': num_reset,
    }


def main():
    parser = argparse.ArgumentParser(description='Train VQ-VAE encoder/decoder model')
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
                       help='Number of warmup steps for learning rate (default: 0, no warmup)')
    parser.add_argument('--input-length', type=int, default=2**16,
                       help='Input audio length in samples (default: 65536)')
    parser.add_argument('--epoch-steps', '--steps', type=int, default=10000,
                       help='Number of training steps per epoch (default: 10000)')
    parser.add_argument('--code-reset-limit', type=int, default=256,
                       help='Codebook reset limit (default: 256)')
    parser.add_argument('--learning-rate', '--lr', type=float, default=2.5e-4,
                       help='Initial learning rate (default: 2e-4)')
    parser.add_argument('--decay-rate', '--half-life', type=float, default=0.5,
                       help='Learning rate decay rate (default: 0.5, halves every decay_steps)')
    parser.add_argument('--decay-steps', type=int, default=None,
                       help='Number of steps for each decay (default: steps * 10)')
    
    args = parser.parse_args()
    
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
    
    print("\nUsing STFT audio loss (multi-scale)")
    
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
    best_loss = float('inf')
    while True:
        start_time = time.time()
        loss_acc = AverageAccumulator()
        audio_loss_acc = AverageAccumulator()
        stft_loss_acc = AverageAccumulator()
        commit_loss_acc = AverageAccumulator()
        nreset = 0
        
        for step in range(args.epoch_steps):
            batch = data.random_batch(args.batch_size, args.input_length)[0]
            result = train_step(encoder, decoder, codebook, opt, restarter, batch)
            
            loss_acc.add(result['loss'])
            audio_loss_acc.add(result['audio_loss'])
            stft_loss_acc.add(result['stft_loss'])
            commit_loss_acc.add(result['commit_loss'])
            nreset += np.sum(result['reset'])
            
            etime = int(args.epoch_steps * ((time.time() - start_time) / (step + 1)))
            etime = '%02d:%02d:%02d' % (etime // 3600, (etime // 60) % 60, etime % 60)
            # Get learning rate
            lr_value = opt.learning_rate
            if callable(lr_value):
                current_lr = float(lr_value(opt.iterations))
            else:
                current_lr = float(lr_value)
            
            print('Epoch=%04d Step=%04d Time=%s LR=%+.4e Loss=%+.4e STFT=%+.4e Commit=%+.4e Used=%05d Reset=%07d  ' % 
                  (epoch, step, etime, current_lr, loss_acc.get(), 
                   stft_loss_acc.get(), commit_loss_acc.get(), np.sum(result['used']), nreset), end='\r')
        print()
        
        # Get average loss for this epoch
        current_loss = loss_acc.get()
        
        # Save weights only if loss improved
        if current_loss < best_loss:
            prev_best = best_loss
            best_loss = current_loss
            model_prefix = args.model
            if prev_best == float('inf'):
                print(f"Saving weights (initial save, loss: {current_loss:.4e})")
            else:
                print(f"Saving weights (loss improved: {current_loss:.4e} < {prev_best:.4e})")
            encoder.save_weights(
                os.path.join(args.output_dir, f'{model_prefix}_encoder.weights.h5')
            )
            decoder.save_weights(
                os.path.join(args.output_dir, f'{model_prefix}_decoder.weights.h5')
            )
            codebook.save_weights(
                os.path.join(args.output_dir, f'{model_prefix}_codebook.weights.h5')
            )
        else:
            print(f"Skipping save (loss did not improve: {current_loss:.4e} >= {best_loss:.4e})")
        
        epoch += 1


if __name__ == '__main__':
    main()

