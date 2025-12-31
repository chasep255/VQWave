#!/usr/bin/env python3
"""
Generate audio from trained generator models.

Supports hierarchical generation with 1-4 levels, using temperature or top-k sampling.
"""

import argparse
import math
import os
import sys

import numpy as np
import tensorflow as tf

from lib.encoder import Encoder, Decoder, CodebookManager
from lib.generator import create_generator
from lib.config import ENCODER_CONFIGS, GENERATOR_CONFIGS, SAMPLE_RATE


@tf.function
def sample_temperature(logits, temperature):
    """Sample using temperature scaling."""
    # logits is 1D [num_codes], need to expand for categorical
    logits_2d = tf.expand_dims(logits / temperature, 0)  # [1, num_codes]
    return tf.random.categorical(logits_2d, 1, dtype=tf.int32)[0, 0]


@tf.function
def sample_top_k(logits, k, temperature=1.0):
    """Sample from top-k logits, optionally with temperature scaling."""
    # logits is 1D [num_codes]
    top_k_logits, top_k_indices = tf.nn.top_k(logits, k)
    # Apply temperature if not 1.0
    if temperature != 1.0:
        top_k_logits = top_k_logits / temperature
    logits_2d = tf.expand_dims(top_k_logits, 0)  # [1, k]
    sampled_idx = tf.random.categorical(logits_2d, 1, dtype=tf.int32)[0, 0]
    return top_k_indices[sampled_idx]


@tf.function
def sample_greedy(logits):
    """Greedy sampling (argmax)."""
    # logits is 1D [num_codes], argmax returns scalar
    return tf.argmax(logits, axis=-1, output_type=tf.int32)


_BASE64_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"


def code_to_ascii(code: int) -> str:
    """
    Render an integer code as a compact ASCII string.

    Uses base64 chars so it stays readable in terminals (unlike random Unicode).
    - For code < 4096: 2 chars
    - Otherwise: variable-length base64 (no padding)
    """
    if code < 0:
        return "??"
    if code < 4096:
        return _BASE64_CHARS[(code >> 6) & 63] + _BASE64_CHARS[code & 63]
    out = []
    x = code
    while x > 0:
        out.append(_BASE64_CHARS[x & 63])
        x >>= 6
    return "".join(reversed(out))


def generate_codes(generator, context_model, num_codes, source_codes, 
                   temperature=None, top_k=None, seed=None, show_codes=False):
    """
    Generate codes autoregressively using a generator.
    
    Args:
        generator: Stateful generator model (batch_size=1)
        context_model: Context model (None if unconditional)
        num_codes: Number of codes to generate
        source_codes: Lower-res codes for context (None if unconditional)
        temperature: Temperature for sampling (None for greedy)
        top_k: Top-k sampling (overrides temperature if set)
        seed: Initial code seed (random if None)
        show_codes: If True, print codes as they're generated
    
    Returns:
        Generated codes as numpy array [num_codes]
    """
    generator.reset_states()
    
    codes = []
    
    # Pre-compute full context if needed (outside generation loop for efficiency)
    context_sequence = None
    if context_model is not None and source_codes is not None:
        # Process all source codes through context model at once
        source_codes_tensor = tf.constant([source_codes], dtype=tf.int32)  # [1, source_len]
        context_sequence = context_model(source_codes_tensor, training=False)  # [1, target_len, context_dim]
        context_sequence = context_sequence[0]  # [target_len, context_dim]
        # Context is upsampled 4x, so it should match or exceed num_codes
        context_len = int(context_sequence.shape[0]) if context_sequence.shape[0] is not None else int(tf.shape(context_sequence)[0].numpy())
    else:
        context_len = 0
    
    # Initial code
    if seed is not None:
        current_code = seed
    else:
        current_code = np.random.randint(0, generator.num_codes)
    
    codes.append(current_code)
    if show_codes:
        print(code_to_ascii(int(current_code)), end='', flush=True)
    
    # Generate remaining codes
    for i in range(num_codes - 1):
        # Prepare input
        input_code = tf.constant([[current_code]], dtype=tf.int32)
        
        # Get context if needed
        if context_sequence is not None:
            # Get context for current position
            context_pos = min(i, context_len - 1)
            context_step = context_sequence[context_pos:context_pos+1]  # [1, context_dim]
            context_step = tf.expand_dims(context_step, 0)  # [1, 1, context_dim]
            logits = generator([input_code, context_step], training=False)
        else:
            logits = generator(input_code, training=False)
        
        # logits shape: [1, 1, num_codes]
        logits = logits[0, 0]  # [num_codes]
        
        # Sample next code
        if top_k is not None:
            # Top-k with optional temperature
            next_code = sample_top_k(logits, top_k, temperature=temperature)
        elif temperature is not None:
            next_code = sample_temperature(logits, temperature)
        else:
            next_code = sample_greedy(logits)
        
        current_code = int(next_code.numpy())
        codes.append(current_code)
        
        if show_codes:
            print(code_to_ascii(int(current_code)), end='', flush=True)
            # New line every 80 codes (i+2 because we already printed first code)
            if (i + 2) % 80 == 0:
                print()
    
    if show_codes:
        print()  # Final newline
    
    return np.array(codes, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser(
        description='Generate audio from trained generator models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 1024 codes using generator_512 from final_weights (no epoch needed)
  %(prog)s --generators generator_512 --length 1024

  # Generate from specific epoch
  %(prog)s --generators generator_512 --epoch 10 --length 1024

  # Generate with temperature sampling
  %(prog)s --generators generator_512 --length 1024 --temperature 0.9

  # Generate with top-k sampling
  %(prog)s --generators generator_512 --length 1024 --top-k 50

  # Generate using all 4 levels hierarchically
  %(prog)s --generators all --length 1024

  # Generate using 2 levels (512 and 128)
  %(prog)s --generators generator_512,generator_128 --length 1024
        """
    )
    
    parser.add_argument('--generators', type=str, required=True,
                       help='Comma-separated generator names or "all" (e.g., "generator_512" or "generator_512,generator_128" or "all")')
    parser.add_argument('--epoch', type=int, default=None,
                       help='Epoch number for generator weights (default: None, loads from final_weights without epoch)')
    parser.add_argument('--length', type=int, required=True,
                       help='Number of codes to generate at the final level (outer codes)')
    parser.add_argument('--temperature', type=float, default=0.9,
                       help='Temperature for sampling (default: 0.9)')
    parser.add_argument('--top-k', type=int, default=None,
                       help='Top-k sampling (overrides temperature if set)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for first code (random if not specified)')
    parser.add_argument('--vqvae-weights-dir', type=str, default='final_weights',
                       help='Directory with VQ-VAE weights (default: final_weights)')
    parser.add_argument('--generator-weights-dir', type=str, default='final_weights',
                       help='Directory with generator weights (default: final_weights)')
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
    
    # Parse generator list
    if args.generators.lower() == 'all':
        generator_names = ['generator_512', 'generator_128', 'generator_32', 'generator_8']
    else:
        requested_names = [g.strip() for g in args.generators.split(',')]
        
        # Validate requested generator names
        for name in requested_names:
            if name not in GENERATOR_CONFIGS:
                print(f"Error: Unknown generator '{name}'. Available: {', '.join(GENERATOR_CONFIGS.keys())}")
                sys.exit(1)
        
        # Determine which level to use (pick the most detailed / lowest compression)
        def get_compression_rate(name):
            return int(GENERATOR_CONFIGS[name]['dest_vqvae'].replace('vqvae_', ''))
        
        final_level = min(requested_names, key=get_compression_rate)  # lowest compression
        
        # Build full chain from final level back to generator_512
        all_levels = ['generator_512', 'generator_128', 'generator_32', 'generator_8']
        final_compression = get_compression_rate(final_level)
        
        # Include all levels up to and including the final level
        generator_names = [level for level in all_levels if get_compression_rate(level) >= final_compression]
        
        if len(requested_names) > 1:
            print(f"Warning: Multiple generators specified. Using full chain ending at {final_level}")
    
    # Sort generators by compression rate (highest first for hierarchical generation)
    def get_compression_rate(name):
        return int(GENERATOR_CONFIGS[name]['dest_vqvae'].replace('vqvae_', ''))
    
    generator_names.sort(key=get_compression_rate, reverse=True)
    
    print(f"Generating with generators: {', '.join(generator_names)}")
    
    # Load VQ-VAE models and generators
    vqvae_models = {}
    generators = {}
    context_models = {}
    
    for gen_name in generator_names:
        gen_config = GENERATOR_CONFIGS[gen_name]
        dest_vqvae_key = gen_config['dest_vqvae']
        source_vqvae_key = gen_config.get('source_vqvae')
        
        # Load destination VQ-VAE decoder (for decoding final codes)
        if dest_vqvae_key not in vqvae_models:
            dest_config = ENCODER_CONFIGS[dest_vqvae_key]
            decoder = Decoder(dest_config)
            codebook = CodebookManager(dest_config)
            
            decoder.load_weights(
                os.path.join(args.vqvae_weights_dir, f'{dest_vqvae_key}_decoder.weights.h5')
            )
            codebook.load_weights(
                os.path.join(args.vqvae_weights_dir, f'{dest_vqvae_key}_codebook.weights.h5')
            )
            
            vqvae_models[dest_vqvae_key] = {'decoder': decoder, 'codebook': codebook}
            print(f"Loaded VQ-VAE: {dest_vqvae_key}")
        
        # Load source VQ-VAE if needed (for context)
        if source_vqvae_key is not None and source_vqvae_key not in vqvae_models:
            source_config = ENCODER_CONFIGS[source_vqvae_key]
            encoder = Encoder(source_config)
            codebook = CodebookManager(source_config)
            
            encoder.load_weights(
                os.path.join(args.vqvae_weights_dir, f'{source_vqvae_key}_encoder.weights.h5')
            )
            codebook.load_weights(
                os.path.join(args.vqvae_weights_dir, f'{source_vqvae_key}_codebook.weights.h5')
            )
            
            vqvae_models[source_vqvae_key] = {'encoder': encoder, 'codebook': codebook}
            print(f"Loaded source VQ-VAE: {source_vqvae_key}")
        
        # Create and load generator
        generator, context_model = create_generator(gen_name, stateful=True, batch_size=1)
        
        # Build weight filenames (with or without epoch)
        if args.epoch is not None:
            generator_weight_file = f'{gen_name}_generator_{args.epoch:05d}.weights.h5'
            context_weight_file = f'{gen_name}_context_{args.epoch:05d}.weights.h5'
        else:
            generator_weight_file = f'{gen_name}_generator.weights.h5'
            context_weight_file = f'{gen_name}_context.weights.h5'
        
        generator.load_weights(
            os.path.join(args.generator_weights_dir, generator_weight_file)
        )
        
        if context_model is not None:
            context_model.load_weights(
                os.path.join(args.generator_weights_dir, context_weight_file)
            )
        
        generators[gen_name] = generator
        context_models[gen_name] = context_model
        print(f"Loaded generator: {gen_name}")
    
    # Determine final compression rate (lowest = most detailed)
    final_gen_name = generator_names[-1]
    final_vqvae_key = GENERATOR_CONFIGS[final_gen_name]['dest_vqvae']
    final_compression = ENCODER_CONFIGS[final_vqvae_key]['compression_rate']
    
    # Length is number of codes at final level
    num_codes = args.length
    actual_audio_length = num_codes * final_compression
    
    print(f"\nGenerating {num_codes} codes at final level")
    print(f"Final compression: {final_compression}x")
    print(f"Audio length: {actual_audio_length} samples ({actual_audio_length / SAMPLE_RATE:.2f} seconds)")
    
    # Generate codes hierarchically
    current_codes = None
    
    for gen_name in generator_names:
        gen_config = GENERATOR_CONFIGS[gen_name]
        dest_vqvae_key = gen_config['dest_vqvae']
        source_vqvae_key = gen_config.get('source_vqvae')
        compression = ENCODER_CONFIGS[dest_vqvae_key]['compression_rate']
        
        # Calculate codes needed for this level
        # For hierarchical generation, each level needs codes proportional to its compression
        # If final level needs num_codes, this level needs: num_codes * (final_compression / compression)
        level_num_codes = math.ceil(num_codes * (final_compression / compression))
        
        print(f"\nGenerating {level_num_codes} codes at {compression}x compression ({gen_name})...")
        
        generator = generators[gen_name]
        context_model = context_models[gen_name]
        
        # Prepare source codes for context if needed
        source_codes_for_context = None
        if source_vqvae_key is not None and current_codes is not None:
            # We have codes from previous level, but we need to encode them to lower-res
            # Actually, current_codes are already at the right level, we just need to use them
            # For context, we need the lower-res codes that correspond to these positions
            # This is a bit tricky - we'd need to decode current_codes to audio, then encode at lower res
            # For now, let's assume we generate hierarchically: higher compression first
            # So current_codes are from the previous (higher compression) level
            source_codes_for_context = current_codes
        
        # Generate codes
        current_codes = generate_codes(
            generator, context_model, level_num_codes,
            source_codes_for_context,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed if gen_name == generator_names[0] else None,
            show_codes=True
        )
        
        print(f"Generated {len(current_codes)} codes, unique: {len(set(current_codes))}")
    
    # Decode final codes to audio
    print(f"\nDecoding {len(current_codes)} codes to audio...")
    final_decoder = vqvae_models[final_vqvae_key]['decoder']
    final_codebook = vqvae_models[final_vqvae_key]['codebook']
    
    # Gather codebook vectors (codes should be [batch, seq_len])
    codes_tensor = tf.expand_dims(tf.constant(current_codes, dtype=tf.int32), 0)  # [1, seq_len]
    code_vectors = final_codebook.gather(codes_tensor)  # [1, seq_len, code_dim]
    
    # Decode to audio
    audio = final_decoder(code_vectors, training=False)
    audio = audio[0].numpy()  # [samples]
    
    # Clip to valid range
    audio = np.clip(audio, -1.0, 1.0)
    
    # Trim to actual length (should match exactly, but trim just in case)
    audio = audio[:actual_audio_length]
    
    print(f"Generated audio: {len(audio)} samples ({len(audio) / SAMPLE_RATE:.2f} seconds)")
    
    # Save or play
    if args.output:
        print(f"Saving audio to: {args.output}")
        from lib.audio import save_audio
        save_audio(args.output, SAMPLE_RATE, audio)
        print("Done!")
    else:
        print("Playing audio...")
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=SAMPLE_RATE,
                output=True
            )
            stream.write(audio.tobytes())
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

