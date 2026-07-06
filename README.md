# VQWave

A Vector Quantized Variational Autoencoder (VQ-VAE) system for music generation.
VQWave compresses audio 256×/512× into a sequence of discrete codes, learns an
autoregressive generator over those codes to synthesize new sequences, and renders
them back to audio with a conditional diffusion decoder.

> **Note**: This project is under active development and may be incomplete. Some features and documentation are still being refined.

## Features

- **256× / 512× Compression**: A VQ-VAE encodes audio into discrete codes. Two presets are provided (`vqvae_256`, `vqvae_512`).
- **Reconstruction Training**: The VQ-VAE is trained with a spectral reconstruction loss (STFT/mel/MSE) plus a VQ commitment loss.
- **Diffusion Rendering Decoder**: A separate, fully convolutional denoising-diffusion model renders a high-fidelity waveform from the codes at generation time (v-prediction, DDIM sampling). Length-agnostic — no attention or positional embeddings.
- **Autoregressive Generation**: A single unconditional generator predicts the code sequence. Two interchangeable architectures are provided (causal Transformer or causal-conv/RNN stack).
- **Efficient Training**: Background-threaded random-crop audio loading for large datasets.
- **Flexible Sampling**: Temperature, top-k, top-p, and greedy sampling.

## Architecture

VQWave has three trained components: a **VQ-VAE** (encoder + codebook + a
deterministic decoder), an autoregressive **generator** over the codes, and a
**diffusion decoder** that renders audio from the codes.

**Training the VQ-VAE (reconstruction):**
```
Audio → Encoder → Codebook (quantize) → Decoder → Reconstructed audio
                                                        │
        Reconstruction loss (STFT/mel/MSE) + VQ commitment loss
```
The deterministic decoder is the "training" decoder: its only job is to shape the
codebook. Final audio is rendered by the diffusion decoder.

**Training the diffusion decoder (codes → waveform):**
```
Audio → (frozen) Encoder + Codebook → integer codes ─┐
                                                      ▼
        noisy waveform  →  conditional U-Net denoiser  →  v-prediction
                          (codes injected at the code rate; time via FiLM)
```

**Generation:**
```
Generator (unconditional, autoregressive)
        ↓
   integer codes
        ↓
   Diffusion decoder (DDIM sampling)   ← or the deterministic VQ-VAE decoder
        ↓
   Audio Output
```

### Components

The VQ-VAE (`vqvae_256` / `vqvae_512`) uses:
- **Encoder**: Convolutional layers that compress audio 256×/512× to latent vectors.
- **Codebook**: Vector quantization with 2048 code vectors (32-dim each).
- **Decoder**: Transposed convolutions that reconstruct audio from quantized codes (training decoder).

The **diffusion decoder** (`diffusion_256` / `diffusion_512`) is a fully
convolutional U-Net denoiser conditioned on the integer codes (embedded and
concatenated at the code rate) and on the diffusion timestep (FiLM). It uses a
continuous-time cosine schedule with v-prediction and deterministic DDIM
sampling. Being fully convolutional, it runs on any audio length that is a
multiple of the compression rate.

The generator predicts the next code autoregressively. Two configs target the
`vqvae_256` codes:
- `generator_256`: causal Transformer with a fixed context window and learned position embeddings.
- `generator_256_rnn`: causal-conv / dilated residual stack (supports stateful step-by-step inference).

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

Training has three stages: (1) train the VQ-VAE, (2) train the diffusion decoder
that renders audio from the VQ-VAE's codes, and (3) train the autoregressive
generator over the codes. Stages 2 and 3 both consume the frozen VQ-VAE and are
independent of each other.

### Stage 1: Train the VQ-VAE

Train the encoder/decoder/codebook using
[`scripts/train_vqvae.py`](scripts/train_vqvae.py):

```bash
python3 scripts/train_vqvae.py \
    --model vqvae_256 \
    --data-dir /path/to/audio/u16/files \
    [--batch-size 8] \
    [--input-length 65536] \
    [--epoch-steps 10000] \
    [--learning-rate 1e-4] \
    [--decay-rate 0.9] \
    [--loss stft] \
    [--commit-weight 0.01] \
    [--output-dir weights] \
    [--bf16] \
    [--warmup-steps 1000]
```

**Training Details:**
- Reconstruction loss is selectable: multi-scale `stft` (default), `mel`, or `mse`.
- The codebook restart mechanism prevents code collapse.
- `--warmup-steps N` runs an Adam *moment* warmup: N steps that settle the optimizer moments (m, v) with weights frozen, so real training starts from a good gradient-variance estimate (this replaces the old LR-ramp warmup).
- `--bf16` enables `mixed_bfloat16` (Ampere+; half the memory, no loss scaling).
- Weights (encoder, decoder, codebook) are saved every epoch to `--output-dir` without epoch numbers, so training is always resumable.

**Resume Training:**
```bash
python3 scripts/train_vqvae.py \
    --model vqvae_256 \
    --data-dir /path/to/data \
    --start-epoch 5
```

### Stage 2: Train the Diffusion Decoder

Train the diffusion decoder using
[`scripts/train_diffusion.py`](scripts/train_diffusion.py). The VQ-VAE encoder +
codebook are frozen and used only to tokenize audio into integer codes; the
denoiser learns to render a waveform from those codes:

```bash
python3 scripts/train_diffusion.py \
    --diffusion diffusion_512 \
    --data-dir /path/to/audio/u16/files \
    --vqvae-weights-dir weights \
    [--batch-size 8] \
    [--input-length 65536] \
    [--epoch-steps 10000] \
    [--learning-rate 1e-4] \
    [--decay-rate 0.9] \
    [--grad-clip 1.0] \
    [--output-dir weights] \
    [--bf16]
```

**Training Details:**
- Pick the `--diffusion` preset whose `dest_vqvae` matches your trained VQ-VAE (`diffusion_256` → `vqvae_256`, `diffusion_512` → `vqvae_512`).
- `--input-length` must be a multiple of the compression rate (the script enforces this).
- Trains with v-prediction (MSE against the v-target) on a continuous-time cosine schedule; timesteps are stratified per batch.
- Waveforms are companded with a mild mu-law (`NORM_MU`, `NORM_STD` in [`vqwave/diffusion.py`](vqwave/diffusion.py)) so the data marginal matches the Gaussian diffusion prior. Re-derive these constants for a very different dataset.
- Weights are saved every epoch as `{diffusion}_denoiser.weights.h5`, without epoch numbers, so training is resumable (`--start-epoch N` or `--load-weights`).
- The deterministic VQ-VAE decoder is **not** used here — only the encoder + codebook.

> **Note:** The diffusion decoder conditions on the codes your encoder currently
> produces. Let the VQ-VAE converge first; if you retrain/change it afterward, the
> codebook shifts and the diffusion decoder must be retrained.

### Stage 3: Train the Generator

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

## Testing the Diffusion Decoder

Render an audio file through its codes with the diffusion decoder using
[`scripts/test_diffusion.py`](scripts/test_diffusion.py). This encodes real audio
to codes with the frozen VQ-VAE, then samples a waveform from those codes — the
same rendering path used at generation time, so it isolates the diffusion
decoder's quality:

```bash
python3 scripts/test_diffusion.py \
    --audio /path/to/audio/file.mp3 \
    --diffusion diffusion_512 \
    [--weights-dir weights] \
    [--steps 200] \
    [--eta 0.0] \
    [--output rendered.wav] \
    [--max-length 30] \
    [--plot] \
    [--no-gpu]
```

**Arguments:**
- `--audio`: Input audio file (mp3, wav, m4a, etc.)
- `--diffusion`: Diffusion config to test (`diffusion_256` or `diffusion_512`)
- `--weights-dir`: Directory with the VQ-VAE and diffusion weights (default: `weights`)
- `--steps`: Number of DDIM sampling steps (default: 200; more = slower, cleaner)
- `--eta`: DDIM stochasticity (0 = deterministic, up to 1 = ancestral)
- `--output`: Save rendered audio to file (default: plays audio)
- `--max-length`: Maximum audio length in seconds (auto-trimmed to a whole number of codes)
- `--plot`: Plot original vs. rendered waveforms
- `--no-gpu`: Run on CPU instead of GPU

Compare this against `test_vqvae.py` on the same clip: the codes are identical,
so any difference is purely deterministic-decoder vs. diffusion-decoder rendering.

## Configuration

Model configurations are defined in [`vqwave/config.py`](vqwave/config.py).

### VQ-VAE Config (`ENCODER_CONFIGS`)

- `vqvae_256`: 256× compression, 2048 codes, 32-dim codebook vectors.
- `vqvae_512`: 512× compression, 2048 codes, 32-dim codebook vectors.

Each preset specifies the encoder layers (convolutional), decoder layers
(transposed convolutional), compression rate (product of encoder strides), and
codebook size/dimension.

### Diffusion Config (`DIFFUSION_CONFIGS`)

- `diffusion_256`: renders `vqvae_256` codes (256× upsample).
- `diffusion_512`: renders `vqvae_512` codes (512× upsample).

Each preset names its `dest_vqvae`, the code-embedding width (`cond_dim`), the
diffusion-time embedding width (`time_dim`), the prediction target (`v` / `eps`),
and `encoder_layers` / `decoder_layers` in the same style as the VQ-VAE. The
encoder strides must multiply to the destination VQ-VAE's compression rate.

### Generator Configs (`GENERATOR_CONFIGS`)

- `generator_256`: Transformer, unconditional, generates 256× codes.
- `generator_256_rnn`: Causal-conv / RNN stack, unconditional, generates 256× codes.

## Project Structure

```
VQWave/
├── vqwave/                 # Core modules
│   ├── encoder.py          # VQ-VAE encoder/decoder/codebook
│   ├── diffusion.py        # Diffusion rendering decoder (codes -> waveform)
│   ├── generator.py        # Transformer and RNN/conv generators
│   ├── audio.py            # Audio loading and processing
│   ├── config.py           # Model configurations
│   ├── layers.py           # Custom layers (codebook, causal conv, etc.)
│   └── util.py             # Utilities (accumulators, LR warmup, etc.)
├── scripts/                # Training and generation scripts
│   ├── prepare_audio.py    # Convert audio to .u16 format
│   ├── train_vqvae.py      # Train VQ-VAE (reconstruction)
│   ├── train_diffusion.py  # Train the diffusion decoder
│   ├── train_generator.py  # Train the generator
│   ├── generate_audio.py   # Generate audio samples
│   ├── test_vqvae.py       # Test VQ-VAE reconstruction
│   └── test_diffusion.py   # Test the diffusion decoder
├── weights/                # Trained model weights
│   ├── vqvae_512_encoder.weights.h5
│   ├── vqvae_512_decoder.weights.h5
│   ├── vqvae_512_codebook.weights.h5
│   ├── diffusion_512_denoiser.weights.h5
│   └── generator_256_generator.weights.h5
└── setup.py                # Package configuration
```

## Troubleshooting

### GPU Memory Issues

- Reduce `--batch-size` if you run out of GPU memory
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

# 2. Train the VQ-VAE
python3 scripts/train_vqvae.py --model vqvae_512 --data-dir /path/to/training/data

# Optional: Test VQ-VAE reconstruction
python3 scripts/test_vqvae.py --audio /path/to/test/audio.mp3 --model vqvae_512 --output reconstructed.wav

# 3. Train the diffusion decoder (VQ-VAE encoder + codebook frozen)
python3 scripts/train_diffusion.py --diffusion diffusion_512 --data-dir /path/to/training/data

# Optional: Render a clip through the codes with the diffusion decoder
python3 scripts/test_diffusion.py --audio /path/to/test/audio.mp3 --diffusion diffusion_512 --output rendered.wav

# 4. Train the generator (VQ-VAE is frozen)
python3 scripts/train_generator.py --generator generator_256 --data-dir /path/to/training/data

# 5. Generate audio
python3 scripts/generate_audio.py \
    --generator generator_256 \
    --length 512 \
    --temperature 0.9 \
    --output generated.wav
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
