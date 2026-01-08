"""
Global configuration constants for audio processing.
"""

# Audio sample rate (Hz)
SAMPLE_RATE = 22050

# VQ-VAE configuration presets
ENCODER_CONFIGS = {
    "vqvae_1024": {
        "compression_rate": 1024,
        "num_codes": 1024,
        "code_dim": 32,
        "encoder_layers": [
            # 4^5 = 1024
            {"channels": 32, "kernel": 12, "stride": 4, "activation": "elu"},
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
            # 4^5 = 1024
            {"channels": 256, "kernel": 12, "stride": 4, "activation": "elu", "transpose": True},
            {"channels": 128, "kernel": 12, "stride": 4, "activation": "elu", "transpose": True},
            {"channels": 64, "kernel": 12, "stride": 4, "activation": "elu", "transpose": True},
            {"channels": 32, "kernel": 12, "stride": 4, "activation": "elu", "transpose": True},
            {"channels": 1, "kernel": 12, "stride": 4, "activation": "tanh", "transpose": True},
        ],
    },

    "vqvae_256": {
        "compression_rate": 256,
        "num_codes": 1024,
        "code_dim": 32,
        "encoder_layers": [
            # 4^4 = 256
            {"channels": 32, "kernel": 12, "stride": 4, "activation": "elu"},
            {"channels": 64, "kernel": 12, "stride": 4, "activation": "elu"},
            {"channels": 128, "kernel": 12, "stride": 4, "activation": "elu"},
            {"channels": 256, "kernel": 12, "stride": 4, "activation": "elu"},
            # Frame-rate refinement
            {"channels": 256, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 256, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 256, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 256, "kernel": 9, "stride": 1, "activation": "elu"},
        ],
        "decoder_layers": [
            # Mirror refinement first
            {"channels": 256, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 256, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 256, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 256, "kernel": 9, "stride": 1, "activation": "elu"},
            # 4^4 = 256
            {"channels": 128, "kernel": 12, "stride": 4, "activation": "elu", "transpose": True},
            {"channels": 64, "kernel": 12, "stride": 4, "activation": "elu", "transpose": True},
            {"channels": 32, "kernel": 12, "stride": 4, "activation": "elu", "transpose": True},
            {"channels": 1, "kernel": 12, "stride": 4, "activation": "tanh", "transpose": True},
        ],
    },

    "vqvae_64": {
        "compression_rate": 64,
        "num_codes": 1024,
        "code_dim": 32,
        "encoder_layers": [
            # 4^3 = 64
            {"channels": 32, "kernel": 12, "stride": 4, "activation": "elu"},
            {"channels": 64, "kernel": 12, "stride": 4, "activation": "elu"},
            {"channels": 128, "kernel": 12, "stride": 4, "activation": "elu"},
            # Frame-rate refinement
            {"channels": 128, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 128, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 128, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 128, "kernel": 9, "stride": 1, "activation": "elu"},

        ],
        "decoder_layers": [
            # Mirror refinement first
            {"channels": 128, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 128, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 128, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 128, "kernel": 9, "stride": 1, "activation": "elu"},

            # 4^3 = 64
            {"channels": 64, "kernel": 12, "stride": 4, "activation": "elu", "transpose": True},
            {"channels": 32, "kernel": 12, "stride": 4, "activation": "elu", "transpose": True},
            {"channels": 1, "kernel": 12, "stride": 4, "activation": "tanh", "transpose": True},
        ],
    },

    "vqvae_16": {
        "compression_rate": 16,
        "num_codes": 1024,
        "code_dim": 32,
        "encoder_layers": [
            # 4^2 = 16
            {"channels": 32, "kernel": 12, "stride": 4, "activation": "elu"},
            {"channels": 64, "kernel": 12, "stride": 4, "activation": "elu"},
            # Frame-rate refinement
            {"channels": 64, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 64, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 64, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 64, "kernel": 9, "stride": 1, "activation": "elu"},
        ],
        "decoder_layers": [
            # Mirror refinement first
            {"channels": 64, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 64, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 64, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 64, "kernel": 9, "stride": 1, "activation": "elu"},
            # 4^2 = 16
            {"channels": 32, "kernel": 12, "stride": 4, "activation": "elu", "transpose": True},
            {"channels": 1, "kernel": 12, "stride": 4, "activation": "tanh", "transpose": True},
        ],
    },
    "vqvae_4": {
        "compression_rate": 4,
        "num_codes": 1024,
        "code_dim": 32,
        "encoder_layers": [
            # 4^1 = 4
            {"channels": 32, "kernel": 12, "stride": 4, "activation": "elu"},

            # Frame-rate refinement
            {"channels": 32, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 32, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 32, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 32, "kernel": 9, "stride": 1, "activation": "elu"},
        ],
        "decoder_layers": [
            # Mirror refinement first
            {"channels": 32, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 32, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 32, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 32, "kernel": 9, "stride": 1, "activation": "elu"},
            # 4^1 = 4
            {"channels": 1, "kernel": 12, "stride": 4, "activation": "tanh", "transpose": True},
        ],
    },
}

# Generator configuration presets
# All parameters are derived from source_vqvae (context) and dest_vqvae (target) configs
GENERATOR_CONFIGS = {
    "generator_1024": {
        "source_vqvae": None,  # Unconditional generation
        "dest_vqvae": "vqvae_1024",  # Generates codes for 1024x compression
        "generator_layers": [
            # Initial projection: embed_dim (32) -> 512 channels
            {"type": "causal_conv", "channels": 512, "kernel": 2, "dilation": 1},
            
            # ResNet blocks with increasing dilation
            {"type": "residual", "channels": 512, "kernel": 2, "dilation": 1, "activation": "elu"},
            {"type": "residual", "channels": 512, "kernel": 2, "dilation": 2, "activation": "elu"},
            {"type": "residual", "channels": 512, "kernel": 2, "dilation": 4, "activation": "elu"},
            {"type": "residual", "channels": 512, "kernel": 2, "dilation": 8, "activation": "elu"},
            {"type": "residual", "channels": 512, "kernel": 2, "dilation": 16, "activation": "elu"},
            {"type": "residual", "channels": 512, "kernel": 2, "dilation": 32, "activation": "elu"},
            {"type": "residual", "channels": 512, "kernel": 2, "dilation": 64, "activation": "elu"},
            {"type": "residual", "channels": 512, "kernel": 2, "dilation": 128, "activation": "elu"},

            {"type": "residual", "channels": 512, "kernel": 2, "dilation": 1, "activation": "elu"},
            {"type": "residual", "channels": 512, "kernel": 2, "dilation": 2, "activation": "elu"},
            {"type": "residual", "channels": 512, "kernel": 2, "dilation": 4, "activation": "elu"},
            {"type": "residual", "channels": 512, "kernel": 2, "dilation": 8, "activation": "elu"},
            {"type": "residual", "channels": 512, "kernel": 2, "dilation": 16, "activation": "elu"},
            {"type": "residual", "channels": 512, "kernel": 2, "dilation": 32, "activation": "elu"},
            {"type": "residual", "channels": 512, "kernel": 2, "dilation": 64, "activation": "elu"},
            {"type": "residual", "channels": 512, "kernel": 2, "dilation": 128, "activation": "elu"},

            {"type": "residual", "channels": 512, "kernel": 2, "dilation": 1, "activation": "elu"},
            {"type": "residual", "channels": 512, "kernel": 2, "dilation": 2, "activation": "elu"},
            {"type": "residual", "channels": 512, "kernel": 2, "dilation": 4, "activation": "elu"},
            {"type": "residual", "channels": 512, "kernel": 2, "dilation": 8, "activation": "elu"},
            {"type": "residual", "channels": 512, "kernel": 2, "dilation": 16, "activation": "elu"},
            {"type": "residual", "channels": 512, "kernel": 2, "dilation": 32, "activation": "elu"},
            {"type": "residual", "channels": 512, "kernel": 2, "dilation": 64, "activation": "elu"},
            {"type": "residual", "channels": 512, "kernel": 2, "dilation": 128, "activation": "elu"},

            {"type": "activation", "activation": "elu"},
            {"type": "conv", "channels": 512, "kernel": 1, "activation": "elu"},
            {"type": "conv", "channels": 512, "kernel": 1, "activation": "elu"},
        ],
    },
    "generator_64": {
        "source_vqvae": "vqvae_1024",  # Context from 1024x codes
        "dest_vqvae": "vqvae_64",  # Generates codes for 64x compression
        "generator_layers": [
            # Initial projection: embed_dim (32) -> 512 channels
            {"type": "context_concat"},
            {"type": "lstm", "units": 512}, 
            {"type": "context_concat"},
            {"type": "conv", "channels": 512, "kernel": 1, "activation": "elu"},
            {"type": "lstm", "units": 512}, 
            {"type": "context_concat"},
            {"type": "conv", "channels": 512, "kernel": 1, "activation": "elu"},
            {"type": "lstm", "units": 512}, 
            {"type": "context_concat"},
            {"type": "conv", "channels": 512, "kernel": 1, "activation": "elu"},
        ],
        # Context model configuration
        "context_layers": [
            # Regular CNN layers for processing
            {"channels": 512, "kernel": 9, "activation": "elu"},
            {"channels": 512, "kernel": 9, "activation": "elu"},
            {"channels": 512, "kernel": 9, "activation": "elu"},
            {"channels": 512, "kernel": 9, "activation": "elu"},
            # Upsample with transpose convolutions (4x * 4x = 16x total)
            {"channels": 256, "kernel": 12, "stride": 4, "transpose": True, "activation": "elu"},
            {"channels": 128, "kernel": 12, "stride": 4, "transpose": True, "activation": "elu"}
        ],
    },
    "generator_16": {
        "source_vqvae": "vqvae_128",  # Context from 128x codes
        "dest_vqvae": "vqvae_16",  # Generates codes for 16x compression
        "generator_layers": [
            {"type": "lstm", "units": 128},
            {"type": "context_concat"},
            {"type": "conv", "channels": 128, "kernel": 1, "activation": "elu"},
            {"type": "lstm", "units": 128},
            {"type": "context_concat"},
            {"type": "conv", "channels": 128, "kernel": 1, "activation": "elu"},
        ],
        # Context model configuration
        "context_layers": [
            # Dilated CNN layers for large receptive field
            {"channels": 256, "kernel": 3, "dilation": 1, "activation": "elu"},
            {"channels": 256, "kernel": 3, "dilation": 2, "activation": "elu"},
            {"channels": 256, "kernel": 3, "dilation": 4, "activation": "elu"},
            {"channels": 256, "kernel": 3, "dilation": 8, "activation": "elu"},
            {"channels": 256, "kernel": 3, "dilation": 16, "activation": "elu"},
            {"channels": 256, "kernel": 3, "dilation": 32, "activation": "elu"},
            # Upsample with transpose convolution (128x -> 16x = 8x)
            {"channels": 128, "kernel": 8, "stride": 8, "activation": "elu", "transpose": True},
        ],
    },
}
