#!/usr/bin/env python3
"""
Test VQ-VAE encoder/decoder by encoding and decoding an audio file.

Loads a trained model, encodes an audio file, decodes it, and plays the result.
"""

import argparse
import os
import sys

import numpy as np
import pyaudio
import tensorflow as tf

from vqwave.encoder import Encoder, Decoder, CodebookManager
from vqwave.config import ENCODER_CONFIGS, SAMPLE_RATE
from vqwave.audio import load_audio, save_audio


def main():
    parser = argparse.ArgumentParser(
        description='Test VQ-VAE encoder/decoder on an audio file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with default settings
  %(prog)s --audio song.mp3 --model vqvae_128 --epoch 100

  # Save output to file instead of playing
  %(prog)s --audio song.mp3 --model vqvae_128 --epoch 100 --output reconstructed.wav

  # Test with different weights directory
  %(prog)s --audio song.mp3 --model vqvae_512 --epoch 50 --weights-dir custom_weights
        """
    )
    
    parser.add_argument('--audio', type=str, required=True,
                       help='Path to input audio file (mp3, wav, m4a, etc.)')
    parser.add_argument('--model', type=str, required=True,
                       choices=['vqvae_512', 'vqvae_128', 'vqvae_32', 'vqvae_8'],
                       help='Model preset name')
    parser.add_argument('--epoch', type=int, required=True,
                       help='Epoch number to load weights from')
    parser.add_argument('--weights-dir', type=str, default='weights',
                       help='Directory containing model weights (default: weights)')
    parser.add_argument('--output', type=str, default=None,
                       help='Save reconstructed audio to file instead of playing')
    parser.add_argument('--max-length', type=int, default=None,
                       help='Maximum audio length in seconds (default: no limit)')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU (use CPU only)')
    
    args = parser.parse_args()
    
    # GPU setup
    if args.no_gpu:
        tf.config.set_visible_devices([], 'GPU')
    else:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    # Validate model preset
    if args.model not in ENCODER_CONFIGS:
        available = ", ".join(ENCODER_CONFIGS.keys())
        print(f"Error: Unknown model preset '{args.model}'. Available: {available}")
        sys.exit(1)
    
    config = ENCODER_CONFIGS[args.model]
    
    # Create models
    print(f"Creating {args.model} model...")
    encoder = Encoder(config)
    decoder = Decoder(config)
    codebook = CodebookManager(config)
    
    # Load weights
    model_prefix = args.model
    weights_dir = args.weights_dir
    epoch_str = f'{args.epoch:05d}'
    
    encoder_path = os.path.join(weights_dir, f'{model_prefix}_encoder_{epoch_str}.weights.h5')
    decoder_path = os.path.join(weights_dir, f'{model_prefix}_decoder_{epoch_str}.weights.h5')
    codebook_path = os.path.join(weights_dir, f'{model_prefix}_codebook_{epoch_str}.weights.h5')
    
    if not os.path.exists(encoder_path):
        print(f"Error: Encoder weights not found: {encoder_path}")
        sys.exit(1)
    if not os.path.exists(decoder_path):
        print(f"Error: Decoder weights not found: {decoder_path}")
        sys.exit(1)
    if not os.path.exists(codebook_path):
        print(f"Error: Codebook weights not found: {codebook_path}")
        sys.exit(1)
    
    print(f"Loading weights from epoch {args.epoch}...")
    encoder.load_weights(encoder_path)
    decoder.load_weights(decoder_path)
    codebook.load_weights(codebook_path)
    
    # Load audio file
    print(f"Loading audio file: {args.audio}")
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        sys.exit(1)
    
    audio = load_audio(args.audio, SAMPLE_RATE)
    original_length = len(audio)
    
    # Limit length if specified
    if args.max_length:
        max_samples = args.max_length * SAMPLE_RATE
        if len(audio) > max_samples:
            audio = audio[:max_samples]
            print(f"Truncated audio to {args.max_length} seconds")
    
    print(f"Audio length: {len(audio) / SAMPLE_RATE:.2f} seconds ({len(audio)} samples)")
    
    # Encode and decode
    print("Encoding audio...")
    audio_tensor = tf.expand_dims(tf.constant(audio, dtype=tf.float32), 0)
    
    # Encode to latents
    z_e = encoder(audio_tensor, training=False)
    
    # Quantize
    z_q, codes = codebook(z_e, training=False)
    
    # Decode
    print("Decoding audio...")
    reconstructed = decoder(z_q, training=False)
    reconstructed_audio = reconstructed[0].numpy()
    
    # Clip to valid range
    reconstructed_audio = np.clip(reconstructed_audio, -1.0, 1.0)
    
    print(f"Reconstructed audio length: {len(reconstructed_audio) / SAMPLE_RATE:.2f} seconds")
    print(f"Unique codes used: {len(set(codes.numpy().flatten()))} / {config['num_codes']}")
    
    # Save or play
    if args.output:
        print(f"Saving reconstructed audio to: {args.output}")
        save_audio(args.output, SAMPLE_RATE, reconstructed_audio)
        print("Done!")
    else:
        print("Playing reconstructed audio...")
        try:
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=SAMPLE_RATE,
                output=True
            )
            stream.write(reconstructed_audio.tobytes())
            stream.stop_stream()
            stream.close()
            p.terminate()
            print("Playback complete!")
        except Exception as e:
            print(f"Error during playback: {e}")
            print("Tip: Install pyaudio or use --output to save to file instead")
            sys.exit(1)


if __name__ == '__main__':
    main()

