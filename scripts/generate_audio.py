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

from vqwave.encoder import Decoder, CodebookManager
from vqwave.generator import create_generator
from vqwave.config import ENCODER_CONFIGS, GENERATOR_CONFIGS, SAMPLE_RATE

# Default values for generation parameters
DEFAULT_TEMPERATURE = 0.9
DEFAULT_TOP_K = 32
DEFAULT_TOP_P = 0.9
DEFAULT_DYNAMIC_TEMPERATURE = False
DEFAULT_ENTROPY_RATIO = 0.65  # 65% of max entropy
DEFAULT_TEMP_BETA = 0.95  # EMA smoothing factor for dynamic temperature
DEFAULT_TEMP_K = 0.05  # Gain for dynamic temperature updates
DEFAULT_TEMP_MIN = 0.7  # Minimum temperature clamp
DEFAULT_TEMP_MAX = 1.1  # Maximum temperature clamp


@tf.function
def sample_temperature(logits, temperature):
    """Sample using temperature scaling."""
    # logits is 1D [num_codes], need to expand for categorical
    logits_2d = tf.expand_dims(logits / tf.cast(temperature, logits.dtype), 0)  # [1, num_codes]
    return tf.random.categorical(logits_2d, 1, dtype=tf.int32)[0, 0]


@tf.function
def sample_top_k(logits, k, temperature=1.0):
    """Sample from top-k logits, optionally with temperature scaling."""
    # logits is 1D [num_codes]
    top_k_logits, top_k_indices = tf.nn.top_k(logits, k)
    # Apply temperature (always divide to avoid Python-side branching in @tf.function)
    top_k_logits = top_k_logits / tf.cast(temperature, top_k_logits.dtype)
    logits_2d = tf.expand_dims(top_k_logits, 0)  # [1, k]
    sampled_idx = tf.random.categorical(logits_2d, 1, dtype=tf.int32)[0, 0]
    return top_k_indices[sampled_idx]


@tf.function
def sample_top_p(logits, p, temperature=1.0):
    """Sample using top-p (nucleus) sampling, optionally with temperature scaling."""
    # logits is 1D [num_codes]
    # Apply temperature (always divide to avoid Python-side branching in @tf.function)
    logits = logits / tf.cast(temperature, logits.dtype)
    
    # Convert to probabilities
    probs = tf.nn.softmax(logits)
    
    # Sort probabilities in descending order
    sorted_probs = tf.sort(probs, direction='DESCENDING')
    sorted_indices = tf.argsort(probs, direction='DESCENDING')
    
    # Compute cumulative probabilities
    cum_probs = tf.cumsum(sorted_probs)
    
    # Find the smallest set where cumulative probability >= p
    # Include tokens where cum_probs <= p, plus the crossing token (the one that exceeds p)
    # This prevents the nucleus from being too small and collapsing to top-1
    vocab_size = tf.shape(logits)[0]
    num_to_keep = tf.reduce_sum(tf.cast(cum_probs <= p, tf.int32)) + 1
    num_to_keep = tf.minimum(num_to_keep, vocab_size)  # Clamp to vocab size
    num_to_keep = tf.maximum(num_to_keep, 1)  # Ensure at least 1 token
    
    # Get top-p logits and indices
    top_p_indices = sorted_indices[:num_to_keep]
    top_p_logits = tf.gather(logits, top_p_indices)
    
    # Sample from top-p set
    logits_2d = tf.expand_dims(top_p_logits, 0)  # [1, num_to_keep]
    sampled_idx = tf.random.categorical(logits_2d, 1, dtype=tf.int32)[0, 0]
    return top_p_indices[sampled_idx]


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


@tf.function
def _entropy_from_logits(logits):
    """Compute entropy from logits."""
    probs = tf.nn.softmax(logits)
    log_probs = tf.math.log(probs + 1e-9)
    return -tf.reduce_sum(probs * log_probs)


def _make_generate_loop(generator, has_context, sampling_mode, temperature, top_k, top_p,
                        dynamic_temp=DEFAULT_DYNAMIC_TEMPERATURE, target_entropy=None,
                        temp_beta=DEFAULT_TEMP_BETA, temp_k=DEFAULT_TEMP_K,
                        temp_min=DEFAULT_TEMP_MIN, temp_max=DEFAULT_TEMP_MAX):
    """
    Create a compiled generation loop function for fast inference.
    
    Returns a tf.function that generates codes in a single graph execution.
    Supports dynamic temperature adjustment within the compiled loop.
    """
    
    @tf.function
    def generate_loop_unconditional(initial_code, num_codes):
        """Fast generation loop without context."""
        codes = tf.TensorArray(dtype=tf.int32, size=num_codes, dynamic_size=False)
        codes = codes.write(0, initial_code)
        current_code = initial_code
        
        # Dynamic temperature state
        T = temperature
        e_ema = tf.constant(0.0, dtype=tf.float32)
        
        for i in tf.range(1, num_codes):
            input_code = tf.reshape(current_code, [1, 1])
            logits = generator(input_code, training=False)
            logits = logits[0, 0]  # [num_codes]
            
            # Update dynamic temperature if enabled
            # This implements a feedback controller to maintain target entropy and prevent repetitive sequences.
            # Algorithm:
            #   1. Compute current entropy H of the predicted distribution (using current temperature T)
            #   2. Calculate error signal: e = target_entropy - H
            #      - If H < target (too repetitive): e > 0 → increase temperature
            #      - If H > target (too random): e < 0 → decrease temperature
            #   3. Smooth error with EMA: e_ema = β * e_ema + (1-β) * e
            #      - Prevents oscillation by averaging recent errors
            #   4. Multiplicative update: T = T * exp(k * e_ema)
            #      - exp(k * e_ema) > 1 when e_ema > 0 (increase T)
            #      - exp(k * e_ema) < 1 when e_ema < 0 (decrease T)
            #      - k controls sensitivity (larger k = faster response)
            #   5. Clamp to bounds: T ∈ [temp_min, temp_max]
            if dynamic_temp:
                H = _entropy_from_logits(logits / T)
                e = target_entropy - H
                e_ema = temp_beta * e_ema + (1.0 - temp_beta) * e
                T = T * tf.exp(temp_k * e_ema)
                T = tf.clip_by_value(T, temp_min, temp_max)
                # Print temperature every step (tf.print returns first tensor, so make T first)
                #tf.print("\nStep", i, "T=", T, "H=", H, "target=", target_entropy)
            
            # Sample based on mode
            if sampling_mode == 'top_p':
                next_code = sample_top_p(logits, top_p, T)
            elif sampling_mode == 'top_k':
                next_code = sample_top_k(logits, top_k, T)
            elif sampling_mode == 'temperature':
                next_code = sample_temperature(logits, T)
            else:
                next_code = sample_greedy(logits)
            
            codes = codes.write(i, next_code)
            current_code = next_code
        
        return codes.stack()
    
    @tf.function
    def generate_loop_with_context(initial_code, num_codes, context_sequence):
        """Fast generation loop with context."""
        codes = tf.TensorArray(dtype=tf.int32, size=num_codes, dynamic_size=False)
        codes = codes.write(0, initial_code)
        current_code = initial_code
        context_len = tf.shape(context_sequence)[0]
        
        # Dynamic temperature state
        T = temperature
        e_ema = tf.constant(0.0, dtype=tf.float32)
        
        for i in tf.range(1, num_codes):
            input_code = tf.reshape(current_code, [1, 1])
            
            # Get context for current position
            context_pos = tf.minimum(i - 1, context_len - 1)
            context_step = context_sequence[context_pos:context_pos+1]  # [1, context_dim]
            context_step = tf.expand_dims(context_step, 0)  # [1, 1, context_dim]
            
            logits = generator([input_code, context_step], training=False)
            logits = logits[0, 0]  # [num_codes]
            
            # Update dynamic temperature if enabled
            # This implements a feedback controller to maintain target entropy and prevent repetitive sequences.
            # Algorithm:
            #   1. Compute current entropy H of the predicted distribution (using current temperature T)
            #   2. Calculate error signal: e = target_entropy - H
            #      - If H < target (too repetitive): e > 0 → increase temperature
            #      - If H > target (too random): e < 0 → decrease temperature
            #   3. Smooth error with EMA: e_ema = β * e_ema + (1-β) * e
            #      - Prevents oscillation by averaging recent errors
            #   4. Multiplicative update: T = T * exp(k * e_ema)
            #      - exp(k * e_ema) > 1 when e_ema > 0 (increase T)
            #      - exp(k * e_ema) < 1 when e_ema < 0 (decrease T)
            #      - k controls sensitivity (larger k = faster response)
            #   5. Clamp to bounds: T ∈ [temp_min, temp_max]
            if dynamic_temp:
                H = _entropy_from_logits(logits / T)
                e = target_entropy - H
                e_ema = temp_beta * e_ema + (1.0 - temp_beta) * e
                T = T * tf.exp(temp_k * e_ema)
                T = tf.clip_by_value(T, temp_min, temp_max)
                # Print temperature every step (tf.print returns first tensor, so make T first)
                #tf.print("\nStep", i, "T=", T, "H=", H, "target=", target_entropy)
            
            # Sample based on mode
            if sampling_mode == 'top_p':
                next_code = sample_top_p(logits, top_p, T)
            elif sampling_mode == 'top_k':
                next_code = sample_top_k(logits, top_k, T)
            elif sampling_mode == 'temperature':
                next_code = sample_temperature(logits, T)
            else:
                next_code = sample_greedy(logits)
            
            codes = codes.write(i, next_code)
            current_code = next_code
        
        return codes.stack()
    
    if has_context:
        return generate_loop_with_context
    else:
        return generate_loop_unconditional


def generate_codes(generator, context_model, num_codes, source_codes, 
                   temperature=None, top_k=None, top_p=None, seed=None, show_codes=False,
                   dynamic_temperature=DEFAULT_DYNAMIC_TEMPERATURE, entropy_ratio=None,
                   temp_beta=DEFAULT_TEMP_BETA, temp_k=DEFAULT_TEMP_K,
                   temp_min=DEFAULT_TEMP_MIN, temp_max=DEFAULT_TEMP_MAX):
    """
    Generate codes autoregressively using a generator.
    
    Args:
        generator: Stateful generator model (batch_size=1)
        context_model: Context model (None if unconditional)
        num_codes: Number of codes to generate
        source_codes: Lower-res codes for context (None if unconditional)
        temperature: Temperature for sampling (None for greedy, or initial temp if dynamic)
        top_k: Top-k sampling (overrides temperature if set)
        top_p: Top-p (nucleus) sampling (overrides top_k and temperature if set)
        seed: Initial code seed (random if None)
        show_codes: If True, print codes as they're generated
        dynamic_temperature: If True, use dynamic temperature that adjusts based on entropy
        entropy_ratio: Target entropy as fraction of max entropy (0.0-1.0, None = auto)
        temp_beta: EMA smoothing factor for dynamic temperature (0.9-0.99)
        temp_k: Gain for dynamic temperature updates (0.01-0.2)
        temp_min: Minimum temperature clamp for dynamic temperature
        temp_max: Maximum temperature clamp for dynamic temperature
    
    Returns:
        Generated codes as numpy array [num_codes]
    """
    generator.reset_states()
    
    # Determine sampling mode and parameters
    if top_p is not None:
        sampling_mode = 'top_p'
        temp = temperature if temperature is not None else 1.0
    elif top_k is not None:
        sampling_mode = 'top_k'
        temp = temperature if temperature is not None else 1.0
    elif temperature is not None:
        sampling_mode = 'temperature'
        temp = temperature
    else:
        sampling_mode = 'greedy'
        temp = 1.0
    
    # Pre-compute full context if needed
    context_sequence = None
    has_context = context_model is not None and source_codes is not None
    if has_context:
        source_codes_tensor = tf.constant([source_codes], dtype=tf.int32)
        context_sequence = context_model(source_codes_tensor, training=False)
        context_sequence = context_sequence[0]  # [target_len, context_dim]
    
    # Initial code
    if seed is not None:
        initial_code = tf.constant(seed, dtype=tf.int32)
    else:
        initial_code = tf.random.uniform([], 0, generator.num_codes, dtype=tf.int32)
    
    # Convert parameters to tensors
    temp_tensor = tf.constant(temp, dtype=tf.float32)
    top_k_tensor = tf.constant(top_k if top_k is not None else DEFAULT_TOP_K, dtype=tf.int32)
    top_p_tensor = tf.constant(top_p if top_p is not None else DEFAULT_TOP_P, dtype=tf.float32)
    
    # Compute target entropy from ratio if dynamic
    if dynamic_temperature:
        if entropy_ratio is None:
            entropy_ratio = DEFAULT_ENTROPY_RATIO
        # Convert ratio to absolute entropy: H_max = log(num_codes), H_target = ratio * H_max
        max_entropy = np.log(generator.num_codes)
        target_entropy = float(max_entropy * entropy_ratio)
        print(f"Dynamic temperature enabled: target_entropy={target_entropy:.4f} (max={max_entropy:.4f}, ratio={entropy_ratio:.2f})")
    else:
        target_entropy = 0.0
    target_entropy_tensor = tf.constant(target_entropy, dtype=tf.float32)
    
    # Get compiled generation function (handles dynamic temperature internally)
    generate_fn = _make_generate_loop(
        generator, has_context, sampling_mode, 
        temp_tensor, top_k_tensor, top_p_tensor,
        dynamic_temp=dynamic_temperature,
        target_entropy=target_entropy_tensor,
        temp_beta=temp_beta,
        temp_k=temp_k,
        temp_min=temp_min,
        temp_max=temp_max
    )
    
    # Run generation
    num_codes_tensor = tf.constant(num_codes, dtype=tf.int32)
    if has_context:
        codes = generate_fn(initial_code, num_codes_tensor, context_sequence)
    else:
        codes = generate_fn(initial_code, num_codes_tensor)
    
    codes = codes.numpy()
    
    # Print codes if requested (after generation for speed)
    if show_codes:
        for i, code in enumerate(codes):
            print(code_to_ascii(int(code)), end='', flush=True)
            if (i + 1) % 80 == 0:
                print()
        print()  # Final newline
    
    return codes


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
        description='Generate audio from trained generator models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate codes using a generator (loads full hierarchy automatically)
  %(prog)s --generator generator_64 --length 1024

  # Generate with temperature sampling (applies to all levels)
  %(prog)s --generator generator_64 --length 1024 --temperature 0.9

  # Generate with different temperatures per level
  %(prog)s --generator generator_64 --length 1024 --temperature 0.8,0.9

  # Generate with top-k sampling
  %(prog)s --generator generator_64 --length 1024 --top-k 50

  # Generate with top-p (nucleus) sampling
  %(prog)s --generator generator_64 --length 1024 --top-p 0.9
        """
    )
    
    parser.add_argument('--generator', type=str, required=True,
                       help=f'Generator name (available: {", ".join(GENERATOR_CONFIGS.keys())}). All prior generators in the hierarchy are automatically loaded.')
    parser.add_argument('--length', type=int, required=True,
                       help='Number of codes to generate at the first/outer layer (highest compression, e.g., 512x). Subsequent levels calculated automatically.')
    parser.add_argument('--temperature', type=str, default=str(DEFAULT_TEMPERATURE),
                       help=f'Temperature for sampling: single value (applies to all levels) or comma-separated values for each level (default: {DEFAULT_TEMPERATURE})')
    parser.add_argument('--top-k', type=int, default=None,
                       help='Top-k sampling (overrides temperature if set)')
    parser.add_argument('--top-p', type=float, default=None,
                       help='Top-p (nucleus) sampling (overrides top-k and temperature if set, typical values: 0.9-0.95)')
    parser.add_argument('--dynamic-temperature', action='store_true',
                       help='Use dynamic temperature that adjusts based on entropy to prevent repetitive sequences')
    parser.add_argument('--entropy-ratio', type=float, default=None,
                       help=f'Target entropy as fraction of max entropy (0.0-1.0, default: {DEFAULT_ENTROPY_RATIO} = {int(DEFAULT_ENTROPY_RATIO*100)}%%)')
    parser.add_argument('--temp-beta', type=float, default=DEFAULT_TEMP_BETA,
                       help=f'EMA smoothing factor for dynamic temperature (0.9-0.99, default: {DEFAULT_TEMP_BETA})')
    parser.add_argument('--temp-k', type=float, default=DEFAULT_TEMP_K,
                       help=f'Gain for dynamic temperature updates (0.01-0.2, default: {DEFAULT_TEMP_K})')
    parser.add_argument('--temp-min', type=float, default=DEFAULT_TEMP_MIN,
                       help=f'Minimum temperature for dynamic temperature (default: {DEFAULT_TEMP_MIN})')
    parser.add_argument('--temp-max', type=float, default=DEFAULT_TEMP_MAX,
                       help=f'Maximum temperature for dynamic temperature (default: {DEFAULT_TEMP_MAX})')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for first code (random if not specified)')
    parser.add_argument('--vqvae-weights-dir', type=str, default='weights',
                       help='Directory with VQ-VAE weights (default: weights)')
    parser.add_argument('--generator-weights-dir', type=str, default='weights',
                       help='Directory with generator weights (default: weights)')
    parser.add_argument('--output', type=str, default=None,
                       help='Save audio to file (optional, otherwise plays)')
    parser.add_argument('--play-intermediates', action='store_true',
                       help='Play intermediate audio at each generation level')
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
    
    # Parse temperature (single value applies to all levels, comma-separated applies per level)
    temperature_str = args.temperature
    if ',' in temperature_str:
        temperatures = [float(t.strip()) for t in temperature_str.split(',')]
    else:
        # Single value - will be used for all levels
        single_temp = float(temperature_str)
        temperatures = [single_temp]
    
    # Parse generator - single generator name, loads full hierarchy
    requested_name = args.generator.strip()
    
    # Validate requested generator
    if requested_name not in GENERATOR_CONFIGS:
        print(f"Error: Unknown generator '{requested_name}'. Available: {', '.join(GENERATOR_CONFIGS.keys())}")
        sys.exit(1)
    
    # Build full hierarchy from requested generator back to highest compression
    def get_compression_rate(name):
        return int(GENERATOR_CONFIGS[name]['dest_vqvae'].replace('vqvae_', ''))
    
    # Get all generators sorted by compression rate (highest first)
    all_levels = sorted(GENERATOR_CONFIGS.keys(), key=get_compression_rate, reverse=True)
    requested_compression = get_compression_rate(requested_name)
    
    # Include all levels up to and including the requested level
    generator_names = [level for level in all_levels if get_compression_rate(level) >= requested_compression]
    
    # Already sorted by compression rate (highest first)
    
    print(f"Generating with generator hierarchy: {', '.join(generator_names)}")
    
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
            print(f"Loaded VQ-VAE decoder: {dest_vqvae_key}")
        
        # Create and load generator
        generator, context_model = create_generator(gen_name, stateful=True, batch_size=1)
        
        # Load weights (always without epoch numbers)
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
        print(f"Loaded generator: {gen_name} from {args.generator_weights_dir}")
    
    # Determine outer (first) and final compression rates
    outer_gen_name = generator_names[0]  # First generator (highest compression, e.g., 512x)
    outer_vqvae_key = GENERATOR_CONFIGS[outer_gen_name]['dest_vqvae']
    outer_compression = ENCODER_CONFIGS[outer_vqvae_key]['compression_rate']
    
    final_gen_name = generator_names[-1]  # Last generator (lowest compression, most detailed, e.g., 8x)
    final_vqvae_key = GENERATOR_CONFIGS[final_gen_name]['dest_vqvae']
    final_compression = ENCODER_CONFIGS[final_vqvae_key]['compression_rate']
    
    # Length always refers to the first/outer layer codes (highest compression)
    num_codes = args.length  # This is the number of codes at the first/outer layer
    actual_audio_length = num_codes * outer_compression
    
    print(f"\nGenerating {num_codes} codes at first/outer layer ({outer_compression}x compression)")
    print(f"Final layer: {final_compression}x compression")
    print(f"Audio length: {actual_audio_length} samples ({actual_audio_length / SAMPLE_RATE:.2f} seconds)")
    
    # Generate codes hierarchically
    # num_codes always refers to the first/outer layer (highest compression)
    current_codes = None
    
    for gen_name in generator_names:
        gen_config = GENERATOR_CONFIGS[gen_name]
        dest_vqvae_key = gen_config['dest_vqvae']
        source_vqvae_key = gen_config.get('source_vqvae')
        compression = ENCODER_CONFIGS[dest_vqvae_key]['compression_rate']
        
        # Calculate codes needed for this level
        # For hierarchical generation, each level needs codes proportional to its compression
        # If outer layer has num_codes, this level needs: num_codes * (outer_compression / compression)
        level_num_codes = math.ceil(num_codes * (outer_compression / compression))
        
        print(f"\nGenerating {level_num_codes} codes at {compression}x compression ({gen_name})...")
        
        generator = generators[gen_name]
        context_model = context_models[gen_name]
        
        # Use codes from previous level as context (if this generator is conditional)
        # Hierarchy runs highest compression first, so current_codes are already at source resolution
        source_codes_for_context = current_codes if source_vqvae_key is not None else None
        
        # Get temperature for this level
        # If fewer temperatures than levels, last value is reused
        level_idx = generator_names.index(gen_name)
        level_temp = temperatures[level_idx] if level_idx < len(temperatures) else temperatures[-1]
        
        # Generate codes
        current_codes = generate_codes(
            generator, context_model, level_num_codes,
            source_codes_for_context,
            temperature=level_temp,
            dynamic_temperature=args.dynamic_temperature,
            entropy_ratio=args.entropy_ratio,
            temp_beta=args.temp_beta,
            temp_k=args.temp_k,
            temp_min=args.temp_min,
            temp_max=args.temp_max,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed if gen_name == generator_names[0] else None,
            show_codes=True
        )
        
        print(f"Generated {len(current_codes)} codes, unique: {len(set(current_codes))}")
        
        # Play intermediate audio if requested (skip last layer, it will be played at the end)
        is_last_layer = (gen_name == generator_names[-1])
        if args.play_intermediates and not is_last_layer:
            print(f"Decoding intermediate audio at {compression}x compression...")
            decoder = vqvae_models[dest_vqvae_key]['decoder']
            codebook = vqvae_models[dest_vqvae_key]['codebook']
            
            # Decode current codes to audio
            codes_tensor = tf.expand_dims(tf.constant(current_codes, dtype=tf.int32), 0)
            code_vectors = codebook.gather(codes_tensor)
            intermediate_audio = decoder(code_vectors, training=False)
            intermediate_audio = intermediate_audio[0].numpy()
            intermediate_audio = np.clip(intermediate_audio, -1.0, 1.0)
            
            # Trim to match expected length for this level
            expected_length = len(current_codes) * compression
            intermediate_audio = intermediate_audio[:expected_length]
            
            print(f"Playing intermediate audio: {len(intermediate_audio)} samples ({len(intermediate_audio) / SAMPLE_RATE:.2f} seconds)")
            try:
                play_audio(intermediate_audio, SAMPLE_RATE)
            except Exception as e:
                print(f"Error during intermediate playback: {e}")
    
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

