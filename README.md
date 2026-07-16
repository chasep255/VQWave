# VQWave

A Vector Quantized Variational Autoencoder (VQ-VAE) system for music generation.
VQWave compresses audio 256×/512× into a sequence of discrete codes, learns an
autoregressive generator over those codes to synthesize new sequences, and decodes
them back to audio. The VQ-VAE is trained with a spectral reconstruction loss plus
an adversarial (WGAN-GP) loss from a waveform critic.

> **Note**: This project is under active development and may be incomplete. Some features and documentation are still being refined.

## Features

- **256× / 512× Compression**: A VQ-VAE encodes audio into discrete codes. Two presets are provided (`vqvae_256`, `vqvae_512`).
- **Adversarial (WGAN-GP) Training**: A waveform critic is trained alongside the VQ-VAE under the WGAN-GP objective, pushing reconstructions toward realistic audio. The spectral reconstruction loss (STFT/mel/MSE) plus the VQ commitment loss remain the anchor tying the codes to the input; the critic only adds realism. Enabled by default (`--adv-weight`).
- **Autoregressive Generation**: A single unconditional generator predicts the code sequence. Interchangeable architectures are provided (causal Transformer or RNN / causal-conv stack).
- **Efficient Training**: Background-threaded random-crop audio loading for large datasets.
- **Flexible Sampling**: Temperature, top-k, top-p, and greedy sampling.

## Architecture

VQWave has two core trained components — a **VQ-VAE** (encoder + codebook +
decoder, trained adversarially against a **critic**) and an autoregressive
**generator** over the codes.

**Training the VQ-VAE (reconstruction + WGAN-GP):**
```
Audio → Encoder → Codebook (quantize) → Decoder → Reconstructed audio
                                                        │
        Reconstruction loss (STFT/mel/MSE) + VQ commitment loss
                              +
        adversarial loss  ←  Critic (WGAN-GP: real vs reconstruction)
```
The critic is training-only; it is never used at inference. It scores a waveform
for realism, and the decoder is pushed to raise that score on its output. The
reconstruction loss stays as the anchor tying codes to the input — see
[Stage 1](#stage-1-train-the-vq-vae).

**Generation:**
```
Generator (unconditional, autoregressive)
        ↓
   integer codes
        ↓
   VQ-VAE decoder
        ↓
   Audio Output
```

### Components

The VQ-VAE (`vqvae_256` / `vqvae_512`) uses:
- **Encoder**: Convolutional layers that compress audio 256×/512× to latent vectors.
- **Codebook**: Vector quantization with 2048 code vectors (32-dim each).
- **Decoder**: Transposed convolutions that reconstruct audio from quantized codes.
- **Critic** (training only, [`vqwave/critic.py`](vqwave/critic.py)): strided convolutions that reduce a waveform to a single unbounded score. It has no normalization layers by design — BatchNorm would couple examples within a batch and invalidate WGAN-GP's per-example gradient penalty. Configured per preset via the `critic` block in [`vqwave/config.py`](vqwave/config.py).

The generator predicts the next code autoregressively. Each preset names its
target VQ-VAE via `dest_vqvae`:
- `generator_256`: causal Transformer, fixed context window with learned position embeddings (targets `vqvae_256`).
- `generator_256_rnn`: causal-conv / dilated residual stack, supports stateful step-by-step inference (targets `vqvae_256`).
- `generator_512_transformer`: causal Transformer (targets `vqvae_512`).
- `generator_512_lstm`: stacked LSTM, supports stateful inference (targets `vqvae_512`).

## Installation

### Requirements

- Python >= 3.8
- CUDA-capable GPU (recommended)
- ffmpeg (for audio processing)
- **SSD for training data** (highly recommended)

### Setup

1. Clone the repository:
```bash
git clone git@github.com:chasep255/VQWave.git
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

### Building a Dataset with spotdl

You can download audio from Spotify playlists, albums, or artists using [spotdl](https://github.com/spotDL/spotify-downloader):

```bash
# Install spotdl
pip install spotdl

# Download a playlist
spotdl download https://open.spotify.com/playlist/37i9dQZF1DX3Kdv0IChEm9 \
  --format m4a \
  --output "{artists} - {title}.{output-ext}" \
  --overwrite skip \
  --threads 8
```

**Query formats:**
- Playlist URL: `https://open.spotify.com/playlist/...`
- Album URL: `https://open.spotify.com/album/...`
- Artist URL: `https://open.spotify.com/artist/...`
- Single track URL: `https://open.spotify.com/track/...`
- Search queries: `album:album name`, `playlist:playlist name`, `artist:artist name`
- Liked songs: use `saved` as the query

**Options:**
- `--format m4a`: Download as M4A format (options: `mp3`, `flac`, `ogg`, `opus`, `m4a`, `wav`)
- `--output "{artists} - {title}.{output-ext}"`: Customize filename format using template variables
- `--overwrite skip`: Skip files that already exist (options: `force`, `metadata`, `skip`)
- `--threads 8`: Use 8 parallel download threads

After downloading, use the conversion script below to prepare the audio for training.

### Supported Formats

- `.m4a`, `.mp3`, `.wav`, `.flac`, `.ogg`

### Conversion

Use [`scripts/prepare_audio.py`](scripts/prepare_audio.py) to convert audio files:

```bash
python3 scripts/prepare_audio.py <source_dir> <dest_dir> [--sample-rate 22050] [--with-meta]
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
- With `--with-meta`: Extracts metadata (artist, title, album, etc.) and saves as `{filename}.meta.json` files

**Note**: Files are processed using ffmpeg. Ensure all input files are valid and accessible.

## Training

Training has two stages: (1) train the VQ-VAE (with its critic), and (2) train
the autoregressive generator over the frozen VQ-VAE's codes.

### Stage 1: Train the VQ-VAE

Train the encoder/decoder/codebook (and the critic) using
[`scripts/train_vqvae.py`](scripts/train_vqvae.py):

```bash
python3 scripts/train_vqvae.py \
    --model vqvae_256 \
    --data-dir /path/to/audio/u16/files \
    [--batch-size 8] \
    [--tokens 256] \
    [--epoch-steps 10000] \
    [--learning-rate 1e-4] \
    [--decay-rate 0.9] \
    [--loss stft] \
    [--commit-weight 0.01] \
    [--adv-weight 0.01] \
    [--gp-weight 10.0] \
    [--n-critic 5] \
    [--critic-lr 1e-4] \
    [--output-dir weights] \
    [--bf16] \
    [--warmup-steps 1000]
```

**Training Details:**
- `--tokens N` sets the number of codebook tokens per training crop; the audio crop length is `tokens * compression_rate` (so 256 tokens = 65536 samples at 256×).
- Reconstruction loss is selectable: multi-scale `stft` (default), `mel`, or `mse`.
- The codebook restart mechanism prevents code collapse.
- `--warmup-steps N` runs an Adam *moment* warmup: N steps that settle the optimizer moments (m, v) with weights frozen, so real training starts from a good gradient-variance estimate (this replaces the old LR-ramp warmup).
- `--bf16` enables `mixed_bfloat16` (Ampere+; half the memory, no loss scaling).
- Weights (encoder, decoder, codebook, and the critic when enabled) are saved every epoch to `--output-dir` without epoch numbers, so training is always resumable.

**Adversarial (WGAN-GP) Training:**
- `--adv-weight` (default `0.01`) weights the adversarial loss. `0` disables the GAN entirely — the critic is not built, trained, or saved, and training reduces to plain reconstruction + commitment.
- `--n-critic` (default `5`) critic updates run per generator update, each on its own batch. This is the WGAN-GP paper value; because the reconstruction loss already anchors the codes here, `--n-critic 1` often suffices and is substantially faster (each critic step needs second-order gradients for the penalty).
- `--gp-weight` (default `10.0`) weights the gradient penalty enforcing the critic's 1-Lipschitz constraint.
- The critic's LR decay is scaled by `--n-critic` so that it and the generator anneal at the same rate per epoch (the critic's optimizer steps `n_critic` times more often).
- The log reports `LR`/`CLR` (generator/critic learning rates), `W` (the Wasserstein estimate the critic maximizes) and `GP`. `GP` starts near 1 (gradient norms are ~0 at init) and should settle toward 0. If `W` runs away, raise `--n-critic` or lower `--critic-lr`.
- The critic is **training-only** and is never loaded at inference, so reconstruction/generation are unaffected by these flags.

**Resume Training:**
```bash
python3 scripts/train_vqvae.py \
    --model vqvae_256 \
    --data-dir /path/to/data \
    --start-epoch 5
```

### Stage 2: Train the Generator

Train the autoregressive generator using
[`scripts/train_generator.py`](scripts/train_generator.py). The VQ-VAE is frozen
and only used to produce the target codes:

```bash
python3 scripts/train_generator.py \
    --generator generator_256 \
    --data-dir /path/to/audio/u16/files \
    --vqvae-weights-dir weights \
    [--batch-size 8] \
    [--input-length 65536] \
    [--epoch-steps 10000] \
    [--learning-rate 1e-3] \
    [--warmup-steps 1000]
```

Choose the architecture by config name:
```bash
# Transformer generator
python3 scripts/train_generator.py --generator generator_256 --data-dir /path/to/data

# RNN / causal-conv generator
python3 scripts/train_generator.py --generator generator_256_rnn --data-dir /path/to/data
```

**Training Details:**
- The generator predicts the next code with sparse categorical crossentropy loss.
- The VQ-VAE encoder/codebook are frozen.
- Gradient clipping (clipnorm=1.0) is applied automatically.
- Weights are saved when the loss improves (best-loss tracking), without epoch numbers.

## Generation

Generate audio using a trained generator + VQ-VAE with
[`scripts/generate_audio.py`](scripts/generate_audio.py):

```bash
python3 scripts/generate_audio.py \
    --generator generator_256 \
    --length 512 \
    [--vqvae-weights-dir weights] \
    [--generator-weights-dir weights] \
    [--temperature 0.9] \
    [--top-k 32] \
    [--top-p 0.9] \
    [--seed <int>] \
    [--output output.wav] \
    [--no-gpu]
```

**Arguments:**
- `--generator`: Generator config name (`generator_256` or `generator_256_rnn`) (required)
- `--length`: Number of codes to generate. Each code spans 256 audio samples (required). For the transformer generator this must be ≤ its `max_seq_len`.
- `--vqvae-weights-dir`: Directory with VQ-VAE weights (default: `weights`)
- `--generator-weights-dir`: Directory with generator weights (default: `weights`)
- `--temperature`: Temperature for sampling (default: `0.9`)
- `--top-k`: Top-k sampling, overrides temperature if set (default: `None`)
- `--top-p`: Top-p (nucleus) sampling, overrides top-k and temperature (default: `None`)
- `--seed`: Initial code seed (default: random)
- `--output`: Save audio to file (default: plays audio)
- `--no-gpu`: Disable GPU (use CPU only)

### Sampling Methods

- **Temperature sampling**: `--temperature 0.9` (lower = more deterministic, higher = more random)
- **Top-k sampling**: `--top-k 32` (samples from the top K most likely codes)
- **Top-p sampling**: `--top-p 0.9` (samples from the smallest set of codes whose cumulative probability exceeds p)
- **Greedy**: omit sampling flags and use `--temperature 0.0`-like behavior via `--top-k 1`

## Testing VQ-VAE

Test VQ-VAE reconstruction quality using [`scripts/test_vqvae.py`](scripts/test_vqvae.py):

```bash
python3 scripts/test_vqvae.py \
    --audio /path/to/audio/file.mp3 \
    --model vqvae_256 \
    [--weights-dir weights] \
    [--output reconstructed.wav] \
    [--max-length 30] \
    [--plot] \
    [--no-gpu]
```

**Arguments:**
- `--audio`: Input audio file (mp3, wav, m4a, etc.)
- `--model`: VQ-VAE model to test (`vqvae_256`)
- `--weights-dir`: Directory containing weights (default: `weights`)
- `--output`: Save reconstructed audio to file (default: plays audio)
- `--max-length`: Maximum audio length in seconds (default: no limit)
- `--plot`: Plot original vs. reconstructed waveforms
- `--no-gpu`: Run on CPU instead of GPU

## Configuration

Model configurations are defined in [`vqwave/config.py`](vqwave/config.py).

### VQ-VAE Config (`ENCODER_CONFIGS`)

- `vqvae_256`: 256× compression, 2048 codes, 32-dim codebook vectors.
- `vqvae_512`: 512× compression, 2048 codes, 32-dim codebook vectors.

Each preset specifies `encoder_layers` (convolutional), `decoder_layers`
(transposed convolutional), `compression_rate` (which must equal the product of
the encoder strides), `num_codes` / `code_dim`, and a `critic` block — the
WGAN-GP critic's strided conv stack (`channels`, `kernel`, `stride`, `alpha`),
used only during training.

### Generator Configs (`GENERATOR_CONFIGS`)

Each preset names its target VQ-VAE via `dest_vqvae`:

- `generator_256`: Transformer, unconditional, generates `vqvae_256` codes.
- `generator_256_rnn`: Causal-conv / RNN stack, unconditional, generates `vqvae_256` codes.
- `generator_512_transformer`: Transformer, unconditional, generates `vqvae_512` codes.
- `generator_512_lstm`: Stacked LSTM, unconditional, generates `vqvae_512` codes.

## Project Structure

```
VQWave/
├── vqwave/                 # Core modules
│   ├── encoder.py          # VQ-VAE encoder/decoder/codebook
│   ├── critic.py           # WGAN-GP critic + gradient penalty (training only)
│   ├── generator.py        # Transformer and RNN/conv generators
│   ├── audio.py            # Audio loading and processing
│   ├── config.py           # Model configurations
│   ├── layers.py           # Custom layers (codebook, causal conv, etc.)
│   └── util.py             # Utilities (accumulators, LR warmup, etc.)
├── scripts/                # Training and generation scripts
│   ├── prepare_audio.py    # Convert audio to .u16 format
│   ├── train_vqvae.py      # Train VQ-VAE (reconstruction + WGAN-GP)
│   ├── train_generator.py  # Train the generator
│   ├── generate_audio.py   # Generate audio samples
│   └── test_vqvae.py       # Test VQ-VAE reconstruction
├── weights/                # Trained model weights
│   ├── vqvae_512_encoder.weights.h5
│   ├── vqvae_512_decoder.weights.h5
│   ├── vqvae_512_codebook.weights.h5
│   └── generator_256_generator.weights.h5
└── setup.py                # Package configuration
```

## Troubleshooting

### GPU Memory Issues

- Reduce `--batch-size` if you run out of GPU memory
- Shorten the training crop: `--tokens` for the VQ-VAE, `--input-length` for the generator
- For the VQ-VAE, `--n-critic 1` cuts the per-step cost substantially (each critic update needs second-order gradients for the penalty), and `--adv-weight 0` disables the critic entirely

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

# 2. Train the VQ-VAE
python3 scripts/train_vqvae.py --model vqvae_512 --data-dir /path/to/training/data

# Optional: Test VQ-VAE reconstruction
python3 scripts/test_vqvae.py --audio /path/to/test/audio.mp3 --model vqvae_512 --output reconstructed.wav

# 3. Train the generator (VQ-VAE is frozen)
python3 scripts/train_generator.py --generator generator_256 --data-dir /path/to/training/data

# 4. Generate audio
python3 scripts/generate_audio.py \
    --generator generator_256 \
    --length 512 \
    --temperature 0.9 \
    --output generated.wav
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
