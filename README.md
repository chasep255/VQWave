# VQWave

A hierarchical Vector Quantized Variational Autoencoder (VQ-VAE) system for music generation. VQWave uses multi-level compression and autoregressive LSTM generators to create audio samples.

> **Note**: This project is under active development and may be incomplete. Some features and documentation are still being refined.

## Features

- **Hierarchical Compression**: Four compression levels (512x, 128x, 32x, 8x) for multi-scale audio representation
- **Autoregressive Generation**: LSTM-based generators that predict code sequences
- **Context Conditioning**: Higher-resolution generators are conditioned on lower-resolution codes
- **Efficient Training**: Memory-mapped audio data for handling large datasets
- **Flexible Sampling**: Temperature, top-k, and greedy sampling methods

## Architecture

VQWave uses a hierarchical approach to audio generation:

**Training (Encoding):**
```
Audio → VQ-VAE Encoders → Quantized Codes
        (512x, 128x, 32x, 8x compression levels)
```

**Generation (Decoding):**
```
Step 1: Generator_512 (unconditional)
        ↓
    512x codes
        ↓
Step 2: Generator_128 (conditioned on 512x codes)
        ↓
    128x codes
        ↓
Step 3: Generator_32 (conditioned on 128x codes)
        ↓
    32x codes
        ↓
Step 4: Generator_8 (conditioned on 32x codes)
        ↓
    8x codes
        ↓
    VQ-VAE Decoder
        ↓
    Audio Output
```

Each generator produces codes at its compression level, which are then used as context for the next level. The final 8x codes are decoded to produce the audio waveform.

### Compression Levels

1. **512x compression** (`vqvae_512`): Coarsest representation, unconditional generation
2. **128x compression** (`vqvae_128`): Conditioned on 512x codes
3. **32x compression** (`vqvae_32`): Conditioned on 128x codes  
4. **8x compression** (`vqvae_8`): Finest representation, conditioned on 32x codes

Each level uses:
- **Encoder**: Convolutional layers that compress audio to latent codes
- **Codebook**: Vector quantization with 1024 code vectors (32-dim each)
- **Decoder**: Transposed convolutions that reconstruct audio from quantized codes
- **Generator**: 2-layer LSTM that predicts next code autoregressively

### Generation Process

Generation proceeds hierarchically:
1. Generate 512x codes unconditionally
2. Generate 128x codes conditioned on 512x codes
3. Generate 32x codes conditioned on 128x codes
4. Generate 8x codes conditioned on 32x codes
5. Decode 8x codes to final audio waveform

## Installation

### Requirements

- Python >= 3.8
- CUDA-capable GPU (recommended)
- ffmpeg (for audio processing)
- **SSD for training data** (highly recommended)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd VQWave
```

2. Run the setup script:
```bash
bash setup.sh
```

This will:
- Create a Python virtual environment
- Install all dependencies (TensorFlow with CUDA, librosa, numpy, matplotlib, tinytag, pyaudio)
- Set up the package in editable mode

**Note**: The setup script checks for PortAudio (required for pyaudio). If missing, install it:
- Ubuntu/Debian: `sudo apt-get install portaudio19-dev`
- Fedora/RHEL: `sudo dnf install portaudio-devel`
- Arch Linux: `sudo pacman -S portaudio`

3. Activate the virtual environment:
```bash
source activate.sh
# or
source venv/bin/activate
```

## Data Preparation

Convert your audio files to the `.u16` format used for training.

### Supported Formats

- `.m4a`, `.mp3`, `.wav`, `.flac`, `.ogg`

### Conversion

Use [`scripts/prepare_audio.py`](scripts/prepare_audio.py) to convert audio files:

```bash
python3 scripts/prepare_audio.py <source_dir> <dest_dir> [--sample-rate 22050]
```

**Example:**
```bash
python3 scripts/prepare_audio.py \
    /path/to/source/audio/files \
    /path/to/destination/u16/files
```

The script:
- Converts all audio files to mono, 22050 Hz sample rate
- Saves as 16-bit unsigned integer (`.u16`) format
- Skips files that already exist in the destination

**Note**: Files are processed using ffmpeg. Ensure all input files are valid and accessible.

## Training

Training is a two-stage process: first train the VQ-VAE models, then train the generators.

### Stage 1: Train VQ-VAE Models

Train encoder/decoder/codebook for each compression level using [`scripts/train_vqvae.py`](scripts/train_vqvae.py):

```bash
python3 scripts/train_vqvae.py \
    --model vqvae_512 \
    --data-dir /path/to/audio/u16/files \
    [--batch-size 8] \
    [--input-length 65536] \
    [--steps 10000] \
    [--output-dir weights] \
    [--fp16]
```

Train each compression level:
```bash
# Train 512x compression
python3 scripts/train_vqvae.py --model vqvae_512 --data-dir /path/to/data

# Train 128x compression  
python3 scripts/train_vqvae.py --model vqvae_128 --data-dir /path/to/data

# Train 32x compression
python3 scripts/train_vqvae.py --model vqvae_32 --data-dir /path/to/data

# Train 8x compression
python3 scripts/train_vqvae.py --model vqvae_8 --data-dir /path/to/data
```

**Training Details:**
- Uses multi-scale STFT loss for reconstruction quality
- Codebook restart mechanism prevents code collapse
- Saves weights after each epoch
- Supports mixed precision training (`--fp16`)

**Resume Training:**
```bash
python3 scripts/train_vqvae.py \
    --model vqvae_512 \
    --data-dir /path/to/data \
    --start-epoch 5  # Resume from epoch 5
```

**After Training:**
Once training is complete, copy your final weights to the `final_weights/` directory (remove the epoch number from the filename):
```bash
cp weights/vqvae_512_encoder_00010.weights.h5 final_weights/vqvae_512_encoder.weights.h5
cp weights/vqvae_512_decoder_00010.weights.h5 final_weights/vqvae_512_decoder.weights.h5
cp weights/vqvae_512_codebook_00010.weights.h5 final_weights/vqvae_512_codebook.weights.h5
# Repeat for vqvae_128, vqvae_32, vqvae_8 with your chosen epoch
```

**Note**: Pre-trained VQ-VAE weights are provided in the `final_weights/` directory. You can skip VQ-VAE training and proceed directly to generator training if desired.

### Stage 2: Train Generators

Train autoregressive generators hierarchically using [`scripts/train_generator.py`](scripts/train_generator.py):

```bash
python3 scripts/train_generator.py \
    --generator generator_512 \
    --data-dir /path/to/audio/u16/files \
    --vqvae-weights-dir final_weights \
    [--batch-size 16] \
    [--input-length 262144] \
    [--steps 10000] \
    [--fp16]
```

Train in order (each depends on the previous):
```bash
# 1. Train unconditional 512x generator
python3 scripts/train_generator.py --generator generator_512 --data-dir /path/to/data

# 2. Train 128x generator (conditioned on 512x)
python3 scripts/train_generator.py --generator generator_128 --data-dir /path/to/data

# 3. Train 32x generator (conditioned on 128x)
python3 scripts/train_generator.py --generator generator_32 --data-dir /path/to/data

# 4. Train 8x generator (conditioned on 32x)
python3 scripts/train_generator.py --generator generator_8 --data-dir /path/to/data
```

**Training Details:**
- Generators predict next code in sequence using 2-layer LSTM
- Context models process lower-res codes with dilated CNNs
- Uses sparse categorical crossentropy loss
- VQ-VAE models are frozen during generator training

**After Training:**
Once generator training is complete, copy your final weights to the `weights/` directory (remove the epoch number from the filename):
```bash
cp weights/generator_512_generator_00010.weights.h5 weights/generator_512_generator.weights.h5
cp weights/generator_512_context_00010.weights.h5 weights/generator_512_context.weights.h5
# Repeat for generator_128, generator_32, generator_8 with your chosen epoch
```

**Note**: Pre-trained generator weights are also provided in the `weights/` directory. You can skip generator training and use my weights.

**Resume Training:**
```bash
python3 scripts/train_generator.py \
    --generator generator_512 \
    --data-dir /path/to/data \
    --start-epoch 4  # Resume from epoch 4
```

## Generation

Generate audio using trained models with [`scripts/generate_audio.py`](scripts/generate_audio.py):

```bash
python3 scripts/generate_audio.py \
    --generators generator_512,generator_128,generator_32,generator_8 \
    --vqvae-weights-dir final_weights \
    --generator-weights-dir weights \
    [--epoch 10] \
    [--length 10000] \
    [--temperature 1.0] \
    [--top-k 0] \
    [--output output.wav]
```

### Sampling Methods

- **Temperature sampling**: `--temperature 1.0` (default)
  - Lower = more deterministic, Higher = more random
  
- **Top-k sampling**: `--top-k 40`
  - Samples from top K most likely codes
  
- **Greedy**: `--top-k 1` or `--temperature 0.0`
  - Always picks most likely code

### Generator Selection

Generate at different quality levels:

```bash
# Low quality (512x only)
python3 scripts/generate_audio.py --generators generator_512

# Medium quality (up to 128x)
python3 scripts/generate_audio.py --generators generator_128

# High quality (up to 32x)
python3 scripts/generate_audio.py --generators generator_32

# Full quality (all levels)
python3 scripts/generate_audio.py --generators generator_8
# or
python3 scripts/generate_audio.py --generators all
```

### Output

- If `--output` is specified, saves to file
- Otherwise, plays audio using pyaudio
- Use `--no-gpu` to run on CPU (slower)

## Configuration

Model configurations are defined in [`lib/config.py`](lib/config.py).

### VQ-VAE Configs

- `vqvae_512`: 512x compression, 1024 codes, 32-dim codebook vectors
- `vqvae_128`: 128x compression, 1024 codes, 32-dim codebook vectors
- `vqvae_32`: 32x compression, 1024 codes, 32-dim codebook vectors
- `vqvae_8`: 8x compression, 1024 codes, 32-dim codebook vectors

Each config specifies:
- Encoder layers (convolutional)
- Decoder layers (transposed convolutional)
- Compression rate (product of encoder strides)
- Codebook size and dimension

### Generator Configs

- `generator_512`: Unconditional, generates 512x codes
- `generator_128`: Conditioned on 512x codes, generates 128x codes
- `generator_32`: Conditioned on 128x codes, generates 32x codes
- `generator_8`: Conditioned on 32x codes, generates 8x codes

Each generator uses:
- 2-layer LSTM (1024 units each)
- Code embeddings (32-dim, matching codebook dimension)
- Optional context model (dilated CNN + upsampling)

## Project Structure

```
VQWave/
├── lib/                    # Core modules
│   ├── encoder.py         # VQ-VAE encoder/decoder/codebook
│   ├── generator.py       # LSTM generators and context models
│   ├── audio.py        # Audio loading and processing
│   ├── config.py          # Model configurations
│   ├── layers.py          # Custom layers (codebook)
│   └── util.py            # Utilities (accumulators, restarters)
├── scripts/               # Training and generation scripts
│   ├── prepare_audio.py   # Convert audio to .u16 format
│   ├── train_vqvae.py     # Train VQ-VAE models
│   ├── train_generator.py # Train generators
│   ├── generate_audio.py # Generate audio samples
│   └── test_vqvae.py     # Test VQ-VAE reconstruction
├── final_weights/         # Pre-trained VQ-VAE weights
│   ├── vqvae_*_encoder.weights.h5
│   ├── vqvae_*_decoder.weights.h5
│   └── vqvae_*_codebook.weights.h5
├── weights/               # Generator weights (training output)
│   ├── generator_*_generator_*.weights.h5
│   └── generator_*_context_*.weights.h5
└── setup.py              # Package configuration
```

## Troubleshooting

### Bus Errors with Memory-Mapped Files

If you encounter "Bus error (core dumped)" during training:

1. **Check for corrupt files**: The error may indicate a corrupted audio file or bad disk sectors
   ```bash
   # Check system logs for I/O errors
   dmesg | grep -i "error\|fail\|bad\|sector" | tail -20
   ```

2. **Filter short files**: Ensure files are long enough for your input length
   - Training scripts automatically filter files shorter than `input_length`
   - Files must be at least `input_length` samples (e.g., 262144 samples ≈ 11.9 seconds at 22050 Hz)

3. **Check disk health**: If errors persist, check for bad sectors on the drive
   - Use `dmesg` to identify problematic sectors
   - Consider replacing the drive if errors are frequent

4. **File descriptor limits**: Already handled automatically in training scripts
   - Scripts increase the file descriptor limit to handle many mmap'd files

### GPU Memory Issues

- Reduce `--batch-size` if you run out of GPU memory
- Use `--fp16` for mixed precision training (reduces memory usage)
- Reduce `--input-length` if needed

### Audio Processing Errors

- Ensure ffmpeg is installed and accessible
- Verify input audio files are valid (not corrupted)
- Check file permissions on source and destination directories

### Import Errors

- Ensure virtual environment is activated
- Install package: `pip install -e .`
- Check Python version: `python3 --version` (needs >= 3.8)

## Example Workflow

Complete example from data preparation to generation:

```bash
# 1. Prepare audio data
python3 scripts/prepare_audio.py \
    /path/to/music/files \
    /path/to/training/data

# 2. Train VQ-VAE models (in parallel or sequentially)
python3 scripts/train_vqvae.py --model vqvae_512 --data-dir /path/to/training/data
python3 scripts/train_vqvae.py --model vqvae_128 --data-dir /path/to/training/data
python3 scripts/train_vqvae.py --model vqvae_32 --data-dir /path/to/training/data
python3 scripts/train_vqvae.py --model vqvae_8 --data-dir /path/to/training/data

# 3. Train generators (sequentially, each depends on previous)
python3 scripts/train_generator.py --generator generator_512 --data-dir /path/to/training/data
python3 scripts/train_generator.py --generator generator_128 --data-dir /path/to/training/data
python3 scripts/train_generator.py --generator generator_32 --data-dir /path/to/training/data
python3 scripts/train_generator.py --generator generator_8 --data-dir /path/to/training/data

# 4. Generate audio
python3 scripts/generate_audio.py \
    --generators all \
    --length 50000 \
    --temperature 1.0 \
    --output generated.wav
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
