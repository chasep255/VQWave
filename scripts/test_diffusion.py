#!/usr/bin/env python3
"""
Test the diffusion decoder by rendering an audio file through the codes.

Loads the frozen VQ-VAE encoder + codebook and a trained diffusion decoder,
tokenizes an audio file into integer codes, then renders a waveform from those
codes via DDIM sampling and plays / saves / plots the result. This is the same
path used at generation time, except the codes come from encoding real audio
instead of from the autoregressive generator -- so it isolates the diffusion
decoder's rendering quality.
"""

import argparse
import os
import sys

import numpy as np
import tensorflow as tf

from vqwave.encoder import Encoder, CodebookManager
from vqwave.diffusion import Denoiser, denormalize
from vqwave.config import DIFFUSION_CONFIGS, ENCODER_CONFIGS, SAMPLE_RATE
from vqwave.audio import load_audio, save_audio


def play_audio(audio, sample_rate):
    """Play audio with Ctrl+C support."""
    import pyaudio

    CHUNK_SIZE = 4096  # Write in chunks for interruptibility
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=sample_rate,
        output=True
    )

    try:
        audio_bytes = audio.astype(np.float32).tobytes()
        for i in range(0, len(audio_bytes), CHUNK_SIZE * 4):  # 4 bytes per float32
            stream.write(audio_bytes[i:i + CHUNK_SIZE * 4])
    except KeyboardInterrupt:
        print("\nPlayback interrupted.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


def main():
    parser = argparse.ArgumentParser(
        description='Test the diffusion decoder by rendering an audio file through its codes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Render a clip through the codes and play it
  %(prog)s --audio song.mp3 --diffusion diffusion_512

  # More sampling steps (slower, higher quality), save to file
  %(prog)s --audio song.mp3 --diffusion diffusion_512 --steps 400 --output out.wav
        """
    )

    parser.add_argument('--audio', type=str, required=True,
                       help='Path to input audio file (mp3, wav, m4a, etc.)')
    parser.add_argument('--diffusion', type=str, required=True,
                       choices=list(DIFFUSION_CONFIGS.keys()),
                       help=f'Diffusion config name (choices: {", ".join(DIFFUSION_CONFIGS.keys())})')
    parser.add_argument('--weights-dir', type=str, default='weights',
                       help='Directory containing VQ-VAE and diffusion weights (default: weights)')
    parser.add_argument('--output', type=str, default=None,
                       help='Save rendered audio to file instead of playing')
    parser.add_argument('--steps', type=int, default=200,
                       help='Number of DDIM sampling steps (default: 200)')
    parser.add_argument('--eta', type=float, default=0.0,
                       help='DDIM stochasticity: 0 = deterministic, up to 1 = ancestral (default: 0.0)')
    parser.add_argument('--max-length', type=int, default=None,
                       help='Maximum audio length in seconds (default: no limit)')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU (use CPU only)')
    parser.add_argument('--plot', action='store_true',
                       help='Plot original and rendered waveforms in a GUI window')

    args = parser.parse_args()

    # GPU setup
    if args.no_gpu:
        tf.config.set_visible_devices([], 'GPU')
    else:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    config = DIFFUSION_CONFIGS[args.diffusion]
    dest_vqvae = config["dest_vqvae"]
    vqvae_config = ENCODER_CONFIGS[dest_vqvae]
    compression = vqvae_config["compression_rate"]
    weights_dir = args.weights_dir

    # Frozen VQ-VAE encoder + codebook (tokenize audio -> codes)
    print(f"Creating frozen VQ-VAE ({dest_vqvae}) and diffusion decoder ({args.diffusion})...")
    encoder = Encoder(vqvae_config)
    codebook = CodebookManager(vqvae_config)
    denoiser = Denoiser(config)

    encoder_path = os.path.join(weights_dir, f'{dest_vqvae}_encoder.weights.h5')
    codebook_path = os.path.join(weights_dir, f'{dest_vqvae}_codebook.weights.h5')
    denoiser_path = os.path.join(weights_dir, f'{args.diffusion}_denoiser.weights.h5')
    for path, what in ((encoder_path, 'encoder'), (codebook_path, 'codebook'),
                       (denoiser_path, 'diffusion denoiser')):
        if not os.path.exists(path):
            print(f"Error: {what} weights not found: {path}")
            sys.exit(1)

    print(f"Loading weights from {weights_dir}...")
    encoder.load_weights(encoder_path)
    codebook.load_weights(codebook_path)
    denoiser.load_weights(denoiser_path)

    # Load audio file
    print(f"Loading audio file: {args.audio}")
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        sys.exit(1)

    audio = load_audio(args.audio, SAMPLE_RATE)

    # Limit length if specified
    if args.max_length:
        max_samples = args.max_length * SAMPLE_RATE
        if len(audio) > max_samples:
            audio = audio[:max_samples]
            print(f"Truncated audio to {args.max_length} seconds")

    # Trim to a whole number of codes (the model needs a multiple of compression).
    usable = (len(audio) // compression) * compression
    if usable == 0:
        print(f"Error: audio is shorter than one code ({compression} samples).")
        sys.exit(1)
    audio = audio[:usable]
    print(f"Audio length: {len(audio) / SAMPLE_RATE:.2f} seconds ({len(audio)} samples)")

    # Tokenize: audio -> integer codes
    print("Encoding audio to codes...")
    audio_tensor = tf.expand_dims(tf.constant(audio, dtype=tf.float32), 0)
    z_e = encoder(audio_tensor, training=False)
    _, codes = codebook(z_e, training=False)
    codes = tf.cast(codes, tf.int32)
    print(f"Codes: {codes.shape[1]} tokens, "
          f"{len(set(codes.numpy().flatten()))} / {vqvae_config['num_codes']} unique")

    # Render: codes -> waveform via DDIM sampling
    print(f"Rendering with diffusion decoder ({args.steps} DDIM steps, eta={args.eta})...")
    normalized = denoiser.generate(codes, nsteps=args.steps, eta=args.eta, progress=True)
    rendered_audio = denormalize(normalized)[0].numpy()
    print(f"Rendered audio length: {len(rendered_audio) / SAMPLE_RATE:.2f} seconds")

    # Plot waveforms if requested
    if args.plot:
        try:
            import matplotlib.pyplot as plt

            min_len = min(len(audio), len(rendered_audio))
            sample_axis = np.arange(min_len)
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            ax.plot(sample_axis, audio[:min_len], 'b-', linewidth=0.5, alpha=0.7, label='Original')
            ax.plot(sample_axis, rendered_audio[:min_len], 'r-', linewidth=0.5, alpha=0.7,
                    label='Diffusion-rendered')
            ax.set_title(f'Diffusion Decoder Rendering ({args.diffusion})',
                         fontsize=16, fontweight='bold')
            ax.set_xlabel('Sample Index', fontsize=12)
            ax.set_ylabel('Amplitude', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-1.1, 1.1)
            ax.legend(loc='upper right', fontsize=11)
            plt.tight_layout()
            print("Displaying waveform plot...")
            plt.show()
        except ImportError:
            print("Error: matplotlib is not installed. Install it with: pip install matplotlib")
            sys.exit(1)
        except Exception as e:
            print(f"Error creating plot: {e}")
            print("Continuing without plot...")

    # Save or play
    if args.output:
        print(f"Saving rendered audio to: {args.output}")
        save_audio(args.output, SAMPLE_RATE, rendered_audio)
        print("Done!")
    else:
        print("Playing rendered audio...")
        try:
            play_audio(rendered_audio, SAMPLE_RATE)
            print("Playback complete!")
        except Exception as e:
            print(f"Error during playback: {e}")
            print("Tip: Install pyaudio or use --output to save to file instead")
            sys.exit(1)


if __name__ == '__main__':
    main()
