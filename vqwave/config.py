"""
Global configuration constants for audio processing.
"""

# Audio sample rate (Hz)
SAMPLE_RATE = 22050

# VQ-VAE configuration presets
#
# Two compression presets are provided (vqvae_256 and vqvae_512), trained with a
# reconstruction loss (see scripts/train_vqvae.py). The deterministic decoder here
# is the "training" decoder that shapes the codebook; a separate diffusion decoder
# (see DIFFUSION_CONFIGS and vqwave/diffusion.py) renders higher-fidelity audio
# from the codes at generation time.
ENCODER_CONFIGS = {
    "vqvae_256": {
        "compression_rate": 256,
        "num_codes": 2048,
        "code_dim": 32,
        "encoder_layers": [
            # 4^4 = 256
            {"channels": 32, "kernel": 12, "stride": 4, "activation": "elu"},
            {"channels": 64, "kernel": 12, "stride": 4, "activation": "elu"},
            {"channels": 128, "kernel": 12, "stride": 4, "activation": "elu"},
            {"channels": 256, "kernel": 12, "stride": 4, "activation": "elu"},
            # Frame-rate refinement
            {"channels": 512, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 512, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 512, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 512, "kernel": 9, "stride": 1, "activation": "elu"},

        ],
        "decoder_layers": [
            # Mirror refinement first
            {"channels": 512, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 512, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 512, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 512, "kernel": 9, "stride": 1, "activation": "elu"},

            # 4^4 = 256
            {"channels": 256, "kernel": 12, "stride": 4, "activation": "elu", "transpose": True},
            {"channels": 128, "kernel": 12, "stride": 4, "activation": "elu", "transpose": True},
            {"channels": 64, "kernel": 12, "stride": 4, "activation": "elu", "transpose": True},
            {"channels": 32, "kernel": 12, "stride": 4, "activation": "elu", "transpose": True},
            # Final projection to a single-channel waveform (Flatten -> [batch, time]).
            {"channels": 1, "kernel": 9, "stride": 1, "activation": "tanh"},
        ],
    },
    "vqvae_512": {
        "compression_rate": 512,
        "num_codes": 2048,
        "code_dim": 32,
        "encoder_layers": [
            # 2 * 4^4 = 512
            {"channels": 32, "kernel": 6, "stride": 2, "activation": "elu"},
            {"channels": 64, "kernel": 12, "stride": 4, "activation": "elu"},
            {"channels": 128, "kernel": 12, "stride": 4, "activation": "elu"},
            {"channels": 256, "kernel": 12, "stride": 4, "activation": "elu"},
            {"channels": 512, "kernel": 12, "stride": 4, "activation": "elu"},
            # Frame-rate refinement
            {"channels": 512, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 512, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 512, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 512, "kernel": 9, "stride": 1, "activation": "elu"},
        ],
        "decoder_layers": [
            # Mirror refinement first
            {"channels": 512, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 512, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 512, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 512, "kernel": 9, "stride": 1, "activation": "elu"},

            # 2 * 4^4 = 512
            {"channels": 256, "kernel": 12, "stride": 4, "activation": "elu", "transpose": True},
            {"channels": 128, "kernel": 12, "stride": 4, "activation": "elu", "transpose": True},
            {"channels": 64, "kernel": 12, "stride": 4, "activation": "elu", "transpose": True},
            {"channels": 32, "kernel": 12, "stride": 4, "activation": "elu", "transpose": True},
            {"channels": 32, "kernel": 6, "stride": 2, "activation": "elu", "transpose": True},
            # Final projection to a single-channel waveform (Flatten -> [batch, time]).
            {"channels": 1, "kernel": 9, "stride": 1, "activation": "tanh"},
        ],
    },
}

# Diffusion decoder configuration presets
#
# A conditional denoising-diffusion model that refines the deterministic VQ-VAE
# reconstruction. The decoded waveform is concatenated channel-wise with the noisy
# input, so the conditioning is present at full resolution and flows through every
# level via the skips -- much stronger than injecting codes at the bottleneck.
#
# Fully convolutional -- no attention or positional embeddings -- so it runs on
# arbitrary-length audio (any multiple of compression_rate). A WaveDiffuse-style
# U-Net contracts then expands:
#   - `stages`: one entry per resolution level, each with two per-stage kernels --
#     `resample_kernel` (the strided down/up conv) and `conv_kernel` (the stride-1
#     refine / skip-fuse conv). Each level is TWO convs and contributes ONE skip
#     connection. The product of the stage strides must equal the destination
#     VQ-VAE's compression_rate.
#   - `middle`: a stack of pre-activated residual dilated blocks (projected into a
#     shared residual width) that widens the receptive field. No attention.
# Every conv is FiLM-conditioned on the diffusion time. Upsampling is a transposed
# conv (keep resample_kernel divisible by stride to avoid checkerboard). A final
# 1-channel projection to the (v / eps) prediction is appended automatically.
DIFFUSION_CONFIGS = {
    "diffusion_256": {
        "dest_vqvae": "vqvae_256",         # refines this VQ-VAE's reconstruction (256x)
        "time_dim": 256,                   # diffusion-time embedding width
        "prediction": "v",                 # "v" (recommended) or "eps"
        # Contract/expand levels; product of strides == 256. Two convs + one skip
        # each: resample_kernel (strided down/up) and conv_kernel (stride-1 refine).
        "stages": [
            {"channels": 64,  "stride": 4, "resample_kernel": 12, "conv_kernel": 9},
            {"channels": 128, "stride": 4, "resample_kernel": 12, "conv_kernel": 9},
            {"channels": 256, "stride": 4, "resample_kernel": 12, "conv_kernel": 9},
            {"channels": 512, "stride": 4, "resample_kernel": 12, "conv_kernel": 9},
        ],
        # Middle: pre-activated residual dilated blocks.
        "middle": [
            {"channels": 512, "kernel": 3, "dilation": 1},
            {"channels": 512, "kernel": 3, "dilation": 2},
            {"channels": 512, "kernel": 3, "dilation": 4},
            {"channels": 512, "kernel": 3, "dilation": 8},
        ],
    },
    "diffusion_512": {
        "dest_vqvae": "vqvae_512",         # refines this VQ-VAE's reconstruction (512x)
        "time_dim": 32,
        "prediction": "v",
        # Contract/expand levels; product of strides == 512. Two convs + one skip
        # each: resample_kernel (strided down/up) and conv_kernel (stride-1 refine).
        "stages": [
            {"channels": 32,  "stride": 2, "resample_kernel": 6,  "conv_kernel": 9},
            {"channels": 64,  "stride": 4, "resample_kernel": 12, "conv_kernel": 9},
            {"channels": 128, "stride": 4, "resample_kernel": 12, "conv_kernel": 9},
            {"channels": 256, "stride": 4, "resample_kernel": 12, "conv_kernel": 9},
            {"channels": 512, "stride": 4, "resample_kernel": 12, "conv_kernel": 9},
        ],
        "middle": [
            {"channels": 512, "kernel": 3, "dilation": 1},
            {"channels": 512, "kernel": 3, "dilation": 2},
            {"channels": 512, "kernel": 3, "dilation": 4},
            {"channels": 512, "kernel": 3, "dilation": 8},
            {"channels": 512, "kernel": 3, "dilation": 16},
            {"channels": 512, "kernel": 3, "dilation": 32},

        ],
    },
}

# Generator configuration presets
#
# A single unconditional generator predicts a VQ-VAE's codes directly. Each preset
# names its target VQ-VAE via `dest_vqvae`. Three architectures are provided:
#   - generator_256:          causal Transformer, fixed context window (targets vqvae_256)
#   - generator_256_rnn:      causal-conv / RNN stack (targets vqvae_256)
#   - generator_512_lstm:     stacked-LSTM stack (targets vqvae_512)
#   - generator_512_transformer: causal Transformer (targets vqvae_512)
GENERATOR_CONFIGS = {
    # Transformer-based generator (fixed context window with position embeddings)
    "generator_256": {
        "type": "transformer",  # Use TransformerGenerator class
        "dest_vqvae": "vqvae_256",  # Generates codes for 256x compression
        "transformer": {
            "max_seq_len": 1024,  # Context window size
            "embedding_dim": 512,  # Residual stream dimension
            "num_layers": 8,  # Number of transformer layers
            "num_heads": 16,  # Number of attention heads
            "key_dim": 64,  # Dimension per attention head
            "ff_dim": 1024,  # Feed-forward hidden dimension
        },
    },
    # RNN / causal-conv generator (autoregressive, supports stateful inference)
    "generator_256_rnn": {
        "type": "rnn",
        "dest_vqvae": "vqvae_256",  # Generates codes for 256x compression
        "generator_layers": [
            # Projection so the residual blocks have a known channel size.
            {"type": "causal_conv", "channels": 512, "kernel": 1, "activation": None},
            # Dilated residual stack for a large receptive field.
            {"type": "residual", "channels": 512, "kernel": 3, "dilation": 1, "activation": "elu"},
            {"type": "residual", "channels": 512, "kernel": 3, "dilation": 2, "activation": "elu"},
            {"type": "residual", "channels": 512, "kernel": 3, "dilation": 4, "activation": "elu"},
            {"type": "residual", "channels": 512, "kernel": 3, "dilation": 8, "activation": "elu"},
            {"type": "residual", "channels": 512, "kernel": 3, "dilation": 16, "activation": "elu"},
            {"type": "residual", "channels": 512, "kernel": 3, "dilation": 32, "activation": "elu"},
            {"type": "residual", "channels": 512, "kernel": 3, "dilation": 1, "activation": "elu"},
            {"type": "residual", "channels": 512, "kernel": 3, "dilation": 2, "activation": "elu"},
            {"type": "residual", "channels": 512, "kernel": 3, "dilation": 4, "activation": "elu"},
            {"type": "residual", "channels": 512, "kernel": 3, "dilation": 8, "activation": "elu"},
            {"type": "residual", "channels": 512, "kernel": 3, "dilation": 16, "activation": "elu"},
            {"type": "residual", "channels": 512, "kernel": 3, "dilation": 32, "activation": "elu"},
            {"type": "activation", "activation": "elu"},
        ],
    },
    # Stacked-LSTM generator targeting the 512x VQ-VAE (supports stateful inference).
    "generator_512_lstm": {
        "type": "rnn",
        "dest_vqvae": "vqvae_512",
        "embedding_dim": 32,
        "generator_layers": [
            {"type": "lstm", "units": 1024},
            {"type": "lstm", "units": 1024},
        ],
    },
    # Transformer generator targeting the 512x VQ-VAE. At 512x compression each
    # code spans twice the audio of the 256x model, so a 1024-code window covers
    # ~24s at 22.05kHz.
    "generator_512_transformer": {
        "type": "transformer",  # Use TransformerGenerator class
        "dest_vqvae": "vqvae_512",  # Generates codes for 512x compression
        "transformer": {
            "max_seq_len": 512,  # Context window size
            "embedding_dim": 512,  # Residual stream dimension
            "num_layers": 12,  # Number of transformer layers
            "num_heads": 8,  # Number of attention heads
            "key_dim": 64,  # Dimension per attention head
            "ff_dim": 1024,  # Feed-forward hidden dimension
        },
    },
}
