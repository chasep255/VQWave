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
# A conditional denoising-diffusion model that renders a waveform from the VQ
# codes -- the high-fidelity "rendering" decoder (the deterministic Decoder is
# only the "training" decoder that shapes the codebook). Conditioned on the
# quantized code embeddings z_q.
#
# Fully convolutional -- no attention or positional embeddings -- so it runs on
# arbitrary-length audio. The layer lists mirror the VQ-VAE style: `encoder_layers`
# downsample the noisy waveform by exactly `compression_rate` (product of the
# encoder strides) down to the code rate, and `decoder_layers` (transpose convs)
# expand it back. The integer codes are embedded (a learned table of width
# `cond_dim`) and concatenated at the code rate (right after the last downsampling
# layer), then the trailing stride-1 encoder layers refine. Every conv is
# FiLM-conditioned on the diffusion time. A final 1-channel projection to the
# (v / eps) prediction is appended automatically.
DIFFUSION_CONFIGS = {
    "diffusion_256": {
        "dest_vqvae": "vqvae_256",         # renders codes for 256x compression
        "cond_dim": 32,                   # code-id embedding width
        "time_dim": 32,                   # diffusion-time embedding width
        "prediction": "v",                 # "v" (recommended) or "eps"
        "encoder_layers": [
            # 4^4 = 256 downsample
            {"channels": 64, "kernel": 12, "stride": 4, "activation": "elu"},
            {"channels": 128, "kernel": 12, "stride": 4, "activation": "elu"},
            {"channels": 256, "kernel": 12, "stride": 4, "activation": "elu"},
            {"channels": 512, "kernel": 12, "stride": 4, "activation": "elu"},
            # code embeddings concatenated here (code rate); refined by:
            {"channels": 512, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 512, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 512, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 512, "kernel": 9, "stride": 1, "activation": "elu"},
        ],
        "decoder_layers": [
            # 4^4 = 256 upsample
            {"channels": 256, "kernel": 12, "stride": 4, "activation": "elu", "transpose": True},
            {"channels": 128, "kernel": 12, "stride": 4, "activation": "elu", "transpose": True},
            {"channels": 64, "kernel": 12, "stride": 4, "activation": "elu", "transpose": True},
            {"channels": 32, "kernel": 12, "stride": 4, "activation": "elu", "transpose": True},
        ],
    },
    "diffusion_512": {
        "dest_vqvae": "vqvae_512",         # renders codes for 512x compression
        "cond_dim": 32,
        "time_dim": 256,
        "prediction": "v",
        "encoder_layers": [
            # 2 * 4^4 = 512 downsample
            {"channels": 64, "kernel": 6, "stride": 2, "activation": "elu"},
            {"channels": 128, "kernel": 12, "stride": 4, "activation": "elu"},
            {"channels": 256, "kernel": 12, "stride": 4, "activation": "elu"},
            {"channels": 256, "kernel": 12, "stride": 4, "activation": "elu"},
            {"channels": 512, "kernel": 12, "stride": 4, "activation": "elu"},
            # code embeddings concatenated here (code rate); refined by:
            {"channels": 512, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 512, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 512, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 512, "kernel": 9, "stride": 1, "activation": "elu"},
        ],
        "decoder_layers": [
            # 2 * 4^4 = 512 upsample
            {"channels": 256, "kernel": 12, "stride": 4, "activation": "elu", "transpose": True},
            {"channels": 256, "kernel": 12, "stride": 4, "activation": "elu", "transpose": True},
            {"channels": 128, "kernel": 12, "stride": 4, "activation": "elu", "transpose": True},
            {"channels": 64, "kernel": 12, "stride": 4, "activation": "elu", "transpose": True},
            {"channels": 32, "kernel": 6, "stride": 2, "activation": "elu", "transpose": True},
        ],
    },
}

# Generator configuration presets
#
# A single unconditional generator predicts the 256x VQ-VAE codes directly.
# Two interchangeable architectures are provided (both target vqvae_256):
#   - generator_256:     causal Transformer (fixed context window)
#   - generator_256_rnn: causal-conv / RNN stack
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
}
