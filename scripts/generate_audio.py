#!/usr/bin/env python3
"""
Generate audio from a trained generator + 256x VQ-VAE.

A single unconditional generator produces a sequence of codes, which the VQ-VAE
decoder turns into a waveform. Supports temperature / top-k / top-p sampling.
"""

import argparse
import os
import sys

import numpy as np
import tensorflow as tf

from vqwave.encoder import Decoder, CodebookManager
from vqwave.generator import create_generator, TransformerGenerator
from vqwave.config import ENCODER_CONFIGS, GENERATOR_CONFIGS, SAMPLE_RATE

# Default values for generation parameters
DEFAULT_TEMPERATURE = 0.9
DEFAULT_TOP_K = 32
DEFAULT_TOP_P = 0.9


@tf.function
def sample_logits(logits, temperature, top_k, top_p, mode):
    """
    Sample from logits using the specified mode.

    Args:
        logits: 1D tensor [num_codes]
        temperature: Temperature for scaling
        top_k: K value for top-k sampling
        top_p: P value for nucleus sampling
        mode: 'greedy', 'temperature', 'top_k', or 'top_p'

    Returns:
        Sampled code as scalar int32
    """
    if mode == 'greedy':
        return tf.argmax(logits, axis=-1, output_type=tf.int32)

    scaled_logits = logits / temperature

    if mode == 'temperature':
        logits_2d = tf.expand_dims(scaled_logits, 0)
        return tf.random.categorical(logits_2d, 1, dtype=tf.int32)[0, 0]

    elif mode == 'top_k':
        top_k_logits, top_k_indices = tf.nn.top_k(scaled_logits, top_k)
        logits_2d = tf.expand_dims(top_k_logits, 0)
        sampled_idx = tf.random.categorical(logits_2d, 1, dtype=tf.int32)[0, 0]
        return top_k_indices[sampled_idx]

    else:  # top_p
        probs = tf.nn.softmax(scaled_logits)
        sorted_probs = tf.sort(probs, direction='DESCENDING')
        sorted_indices = tf.argsort(probs, direction='DESCENDING')
        cum_probs = tf.cumsum(sorted_probs)

        vocab_size = tf.shape(logits)[0]
        num_to_keep = tf.reduce_sum(tf.cast(cum_probs <= top_p, tf.int32)) + 1
        num_to_keep = tf.clip_by_value(num_to_keep, 1, vocab_size)

        top_p_indices = sorted_indices[:num_to_keep]
        top_p_logits = tf.gather(scaled_logits, top_p_indices)

        logits_2d = tf.expand_dims(top_p_logits, 0)
        sampled_idx = tf.random.categorical(logits_2d, 1, dtype=tf.int32)[0, 0]
        return top_p_indices[sampled_idx]


@tf.function
def _codes_to_base64_string(codes):
    """Convert a tensor of codes to a base64-encoded string."""
    base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    chars_tensor = tf.constant(list(base64_chars), dtype=tf.string)

    codes_clamped = tf.minimum(codes, 4095)
    char1_indices = tf.bitwise.bitwise_and(tf.bitwise.right_shift(codes_clamped, 6), 63)
    char2_indices = tf.bitwise.bitwise_and(codes_clamped, 63)

    char1 = tf.gather(chars_tensor, char1_indices)
    char2 = tf.gather(chars_tensor, char2_indices)

    char_pairs = tf.strings.join([char1, char2], separator='')
    return tf.strings.reduce_join(char_pairs, separator='')


def _make_rnn_generate_loop(generator, mode, temperature, top_k, top_p):
    """Create a compiled generation loop for the stateful RNN/conv generator."""

    @tf.function
    def generate_loop(initial_code, num_codes):
        codes = tf.TensorArray(dtype=tf.int32, size=num_codes, dynamic_size=False)
        codes = codes.write(0, initial_code)
        current_code = initial_code

        LINE_LENGTH = 80
        line_buffer = tf.TensorArray(dtype=tf.int32, size=LINE_LENGTH, dynamic_size=False)

        for i in tf.range(1, num_codes):
            input_code = tf.reshape(current_code, [1, 1])
            logits = generator(input_code, training=False)
            logits = logits[0, 0]
            next_code = sample_logits(logits, temperature, top_k, top_p, mode)

            codes = codes.write(i, next_code)

            buffer_pos = i % LINE_LENGTH
            line_buffer = line_buffer.write(buffer_pos, next_code)
            should_print = tf.logical_and(tf.equal(buffer_pos, 0), tf.greater(i, 0))
            if should_print:
                line_codes = line_buffer.gather(tf.range(LINE_LENGTH))
                tf.print(_codes_to_base64_string(line_codes))

            current_code = next_code

        remaining_count = num_codes % LINE_LENGTH
        if remaining_count > 0:
            remaining_codes = line_buffer.gather(tf.range(remaining_count))
            tf.print(_codes_to_base64_string(remaining_codes))

        return codes.stack()

    return generate_loop


def _make_transformer_generate_loop(generator, mode, temperature, top_k, top_p):
    """Create a generation loop for TransformerGenerator (fixed context window)."""

    @tf.function
    def generate_loop(initial_code, num_codes):
        codes = tf.TensorArray(dtype=tf.int32, size=num_codes, dynamic_size=False)
        codes = codes.write(0, initial_code)

        sequence = tf.TensorArray(dtype=tf.int32, size=num_codes, dynamic_size=False)
        sequence = sequence.write(0, initial_code)

        LINE_LENGTH = 80
        line_buffer = tf.TensorArray(dtype=tf.int32, size=LINE_LENGTH, dynamic_size=False)

        for i in tf.range(1, num_codes):
            current_seq = sequence.stack()[:i]
            input_seq = tf.expand_dims(current_seq, 0)  # [1, i]

            logits = generator(input_seq, training=False)
            logits = logits[0, -1]  # Last position logits

            next_code = sample_logits(logits, temperature, top_k, top_p, mode)

            codes = codes.write(i, next_code)
            sequence = sequence.write(i, next_code)

            buffer_pos = i % LINE_LENGTH
            line_buffer = line_buffer.write(buffer_pos, next_code)
            should_print = tf.logical_and(tf.equal(buffer_pos, 0), tf.greater(i, 0))
            if should_print:
                line_codes = line_buffer.gather(tf.range(LINE_LENGTH))
                tf.print(_codes_to_base64_string(line_codes))

        remaining_count = num_codes % LINE_LENGTH
        if remaining_count > 0:
            remaining_codes = line_buffer.gather(tf.range(remaining_count))
            tf.print(_codes_to_base64_string(remaining_codes))

        return codes.stack()

    return generate_loop


def generate_codes(generator, num_codes, temperature=None, top_k=None, top_p=None, seed=None):
    """
    Generate codes autoregressively.

    Args:
        generator: Generator model (stateful RNN/conv or TransformerGenerator)
        num_codes: Number of codes to generate
        temperature: Temperature for sampling (None for greedy)
        top_k: Top-k sampling (overrides temperature if set)
        top_p: Top-p (nucleus) sampling (overrides top_k and temperature if set)
        seed: Initial code seed (random if None)

    Returns:
        Generated codes as numpy array [num_codes]
    """
    generator.reset_states()

    # Determine sampling mode
    if top_p is not None:
        mode = 'top_p'
        temp = temperature if temperature is not None else 1.0
    elif top_k is not None:
        mode = 'top_k'
        temp = temperature if temperature is not None else 1.0
    elif temperature is not None:
        mode = 'temperature'
        temp = temperature
    else:
        mode = 'greedy'
        temp = 1.0

    if seed is not None:
        initial_code = tf.constant(seed, dtype=tf.int32)
    else:
        initial_code = tf.random.uniform([], 0, generator.num_codes, dtype=tf.int32)

    temp_tensor = tf.constant(temp, dtype=tf.float32)
    top_k_tensor = tf.constant(top_k if top_k is not None else DEFAULT_TOP_K, dtype=tf.int32)
    top_p_tensor = tf.constant(top_p if top_p is not None else DEFAULT_TOP_P, dtype=tf.float32)
    num_codes_tensor = tf.constant(num_codes, dtype=tf.int32)

    if isinstance(generator, TransformerGenerator):
        if num_codes > generator.max_seq_len:
            raise ValueError(
                f"Transformer generator can only generate up to {generator.max_seq_len} codes, "
                f"but {num_codes} requested. Use --length {generator.max_seq_len} or less."
            )
        print(f"Using transformer mode (generating {num_codes}/{generator.max_seq_len} codes)")
        generate_fn = _make_transformer_generate_loop(
            generator, mode, temp_tensor, top_k_tensor, top_p_tensor
        )
    else:
        generate_fn = _make_rnn_generate_loop(
            generator, mode, temp_tensor, top_k_tensor, top_p_tensor
        )

    return generate_fn(initial_code, num_codes_tensor).numpy()


def play_audio(audio, sample_rate):
    """Play audio with Ctrl+C support."""
    import pyaudio

    CHUNK_SIZE = 4096
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=sample_rate,
        output=True
    )

    try:
        audio_bytes = audio.astype(np.float32).tobytes()
        for i in range(0, len(audio_bytes), CHUNK_SIZE * 4):
            stream.write(audio_bytes[i:i + CHUNK_SIZE * 4])
    except KeyboardInterrupt:
        print("\nPlayback interrupted.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


def main():
    parser = argparse.ArgumentParser(
        description='Generate audio from a trained generator + 256x VQ-VAE',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 512 codes with the transformer generator
  %(prog)s --generator generator_256 --length 512

  # Temperature sampling
  %(prog)s --generator generator_256 --length 512 --temperature 0.9

  # Top-k / top-p sampling
  %(prog)s --generator generator_256 --length 512 --top-k 32
  %(prog)s --generator generator_256 --length 512 --top-p 0.9
        """
    )

    parser.add_argument('--generator', type=str, required=True,
                       choices=list(GENERATOR_CONFIGS.keys()),
                       help=f'Generator name (available: {", ".join(GENERATOR_CONFIGS.keys())})')
    parser.add_argument('--length', type=int, required=True,
                       help='Number of codes to generate')
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE,
                       help=f'Temperature for sampling (default: {DEFAULT_TEMPERATURE})')
    parser.add_argument('--top-k', type=int, default=None,
                       help='Top-k sampling (overrides temperature if set)')
    parser.add_argument('--top-p', type=float, default=None,
                       help='Top-p (nucleus) sampling (overrides top-k and temperature)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Initial code seed (random if not specified)')
    parser.add_argument('--vqvae-weights-dir', type=str, default='weights',
                       help='Directory with VQ-VAE weights (default: weights)')
    parser.add_argument('--generator-weights-dir', type=str, default='weights',
                       help='Directory with generator weights (default: weights)')
    parser.add_argument('--output', type=str, default=None,
                       help='Save audio to file (optional, otherwise plays)')
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

    gen_config = GENERATOR_CONFIGS[args.generator]
    dest_vqvae_key = gen_config['dest_vqvae']
    vqvae_config = ENCODER_CONFIGS[dest_vqvae_key]
    compression = vqvae_config['compression_rate']

    # Load VQ-VAE decoder + codebook
    decoder = Decoder(vqvae_config)
    codebook = CodebookManager(vqvae_config)
    decoder.load_weights(os.path.join(args.vqvae_weights_dir, f'{dest_vqvae_key}_decoder.weights.h5'))
    codebook.load_weights(os.path.join(args.vqvae_weights_dir, f'{dest_vqvae_key}_codebook.weights.h5'))
    print(f"Loaded VQ-VAE decoder: {dest_vqvae_key} ({compression}x compression)")

    # Create generator
    is_transformer = gen_config.get('type') == 'transformer'
    if is_transformer:
        generator = create_generator(args.generator)
    else:
        generator = create_generator(args.generator, stateful=True, batch_size=1)
    generator.load_weights(
        os.path.join(args.generator_weights_dir, f'{args.generator}_generator.weights.h5')
    )
    print(f"Loaded generator: {args.generator}")

    num_codes = args.length
    audio_length = num_codes * compression
    print(f"\nGenerating {num_codes} codes ({compression}x compression)")
    print(f"Audio length: {audio_length} samples ({audio_length / SAMPLE_RATE:.2f} seconds)")

    sampling_info = (f"top_p={args.top_p}" if args.top_p is not None
                     else f"top_k={args.top_k}" if args.top_k is not None
                     else f"temperature={args.temperature}")
    print(f"Sampling: {sampling_info}\n")

    codes = generate_codes(
        generator, num_codes,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )
    print(f"Generated {len(codes)} codes, unique: {len(set(codes))}")

    # Decode codes to audio
    print(f"\nDecoding {len(codes)} codes to audio...")
    codes_tensor = tf.expand_dims(tf.constant(codes, dtype=tf.int32), 0)
    code_vectors = codebook.gather(codes_tensor)
    audio = decoder(code_vectors, training=False)
    audio = audio[0].numpy()
    audio = np.clip(audio, -1.0, 1.0)
    audio = audio[:audio_length]

    print(f"Generated audio: {len(audio)} samples ({len(audio) / SAMPLE_RATE:.2f} seconds)")

    if args.output:
        print(f"Saving audio to: {args.output}")
        from vqwave.audio import save_audio
        save_audio(args.output, SAMPLE_RATE, audio)
        print("Done!")
    else:
        print("Playing audio...")
        try:
            play_audio(audio, SAMPLE_RATE)
            print("Playback complete!")
        except Exception as e:
            print(f"Error during playback: {e}")
            print("Tip: Install pyaudio or use --output to save to file instead")
            sys.exit(1)


if __name__ == '__main__':
    main()
