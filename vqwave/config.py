"""
Global configuration constants for audio processing.
"""

# Audio sample rate (Hz)
SAMPLE_RATE = 22050

# VQ-VAE configuration presets
ENCODER_CONFIGS = {
    "vqvae_512": {
        "compression_rate": 512,
        "num_codes": 1024,
        "code_dim": 32,
        "encoder_layers": [
            {"channels": 32, "kernel": 6, "stride": 2, "activation": "elu"},
            {"channels": 48, "kernel": 6, "stride": 2, "activation": "elu"},
            {"channels": 64, "kernel": 6, "stride": 2, "activation": "elu"},
            {"channels": 96, "kernel": 6, "stride": 2, "activation": "elu"},
            {"channels": 128, "kernel": 6, "stride": 2, "activation": "elu"},
            {"channels": 192, "kernel": 6, "stride": 2, "activation": "elu"},
            {"channels": 256, "kernel": 6, "stride": 2, "activation": "elu"},
            {"channels": 384, "kernel": 6, "stride": 2, "activation": "elu"},
            {"channels": 512, "kernel": 6, "stride": 2, "activation": "elu"},
            {"channels": 512, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 512, "kernel": 9, "stride": 1, "activation": "elu"},
        ],
        "decoder_layers": [
            {"channels": 512, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 512, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 512, "kernel": 6, "stride": 2, "activation": "elu", "transpose": True},
            {"channels": 384, "kernel": 6, "stride": 2, "activation": "elu", "transpose": True},
            {"channels": 256, "kernel": 6, "stride": 2, "activation": "elu", "transpose": True},
            {"channels": 192, "kernel": 6, "stride": 2, "activation": "elu", "transpose": True},
            {"channels": 128, "kernel": 6, "stride": 2, "activation": "elu", "transpose": True},
            {"channels": 96, "kernel": 6, "stride": 2, "activation": "elu", "transpose": True},
            {"channels": 64, "kernel": 6, "stride": 2, "activation": "elu", "transpose": True},
            {"channels": 48, "kernel": 6, "stride": 2, "activation": "elu", "transpose": True},
            {"channels": 32, "kernel": 6, "stride": 2, "activation": "elu", "transpose": True},
        ],
    },
    "vqvae_128": {
        "compression_rate": 128,
        "num_codes": 1024,
        "code_dim": 32,
        "encoder_layers": [
            {"channels": 32, "kernel": 6, "stride": 2, "activation": "elu"},
            {"channels": 48, "kernel": 6, "stride": 2, "activation": "elu"},
            {"channels": 64, "kernel": 6, "stride": 2, "activation": "elu"},
            {"channels": 96, "kernel": 6, "stride": 2, "activation": "elu"},
            {"channels": 128, "kernel": 6, "stride": 2, "activation": "elu"},
            {"channels": 192, "kernel": 6, "stride": 2, "activation": "elu"},
            {"channels": 256, "kernel": 6, "stride": 2, "activation": "elu"},
            {"channels": 384, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 384, "kernel": 9, "stride": 1, "activation": "elu"},
        ],
        "decoder_layers": [
            {"channels": 384, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 384, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 256, "kernel": 6, "stride": 2, "activation": "elu", "transpose": True},
            {"channels": 192, "kernel": 6, "stride": 2, "activation": "elu", "transpose": True},
            {"channels": 128, "kernel": 6, "stride": 2, "activation": "elu", "transpose": True},
            {"channels": 96, "kernel": 6, "stride": 2, "activation": "elu", "transpose": True},
            {"channels": 64, "kernel": 6, "stride": 2, "activation": "elu", "transpose": True},
            {"channels": 48, "kernel": 6, "stride": 2, "activation": "elu", "transpose": True},
            {"channels": 32, "kernel": 6, "stride": 2, "activation": "elu", "transpose": True},
        ],
    },
    "vqvae_32": {
        "compression_rate": 32,
        "num_codes": 1024,
        "code_dim": 32,
        "encoder_layers": [
            {"channels": 32, "kernel": 6, "stride": 2, "activation": "elu"},
            {"channels": 48, "kernel": 6, "stride": 2, "activation": "elu"},
            {"channels": 64, "kernel": 6, "stride": 2, "activation": "elu"},
            {"channels": 96, "kernel": 6, "stride": 2, "activation": "elu"},
            {"channels": 128, "kernel": 6, "stride": 2, "activation": "elu"},
            {"channels": 192, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 192, "kernel": 9, "stride": 1, "activation": "elu"},
        ],
        "decoder_layers": [
            {"channels": 192, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 192, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 128, "kernel": 6, "stride": 2, "activation": "elu", "transpose": True},
            {"channels": 96, "kernel": 6, "stride": 2, "activation": "elu", "transpose": True},
            {"channels": 64, "kernel": 6, "stride": 2, "activation": "elu", "transpose": True},
            {"channels": 48, "kernel": 6, "stride": 2, "activation": "elu", "transpose": True},
            {"channels": 32, "kernel": 6, "stride": 2, "activation": "elu", "transpose": True},
        ],
    },
    "vqvae_8": {
        "compression_rate": 8,
        "num_codes": 1024,
        "code_dim": 32,
        "encoder_layers": [
            {"channels": 32, "kernel": 6, "stride": 2, "activation": "elu"},
            {"channels": 48, "kernel": 6, "stride": 2, "activation": "elu"},
            {"channels": 64, "kernel": 6, "stride": 2, "activation": "elu"},
            {"channels": 96, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 96, "kernel": 9, "stride": 1, "activation": "elu"},
        ],
        "decoder_layers": [
            {"channels": 96, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 96, "kernel": 9, "stride": 1, "activation": "elu"},
            {"channels": 64, "kernel": 6, "stride": 2, "activation": "elu", "transpose": True},
            {"channels": 48, "kernel": 6, "stride": 2, "activation": "elu", "transpose": True},
            {"channels": 32, "kernel": 6, "stride": 2, "activation": "elu", "transpose": True},
        ],
    },
}

# Generator configuration presets
# All parameters are derived from source_vqvae (context) and dest_vqvae (target) configs
GENERATOR_CONFIGS = {
    "generator_512": {
        "source_vqvae": None,  # Unconditional generation
        "dest_vqvae": "vqvae_512",  # Generates codes for 512x compression
        "lstm_units": 512,
        "lstm_layers": 2,
    },
    "generator_128": {
        "source_vqvae": "vqvae_512",  # Context from 512x codes
        "dest_vqvae": "vqvae_128",  # Generates codes for 128x compression
        "lstm_units": 512,
        "lstm_layers": 2,
        # Context model configuration
        "context_dim": 512,  # Output dimension of context features
        "context_channels": 512,  # Intermediate channels in context model dilated CNN
        "context_dilations": [1, 2, 4, 8, 16, 32],  # Dilation rates for each layer
        "context_kernel_size": 3,  # Kernel size for dilated conv layers
        "context_activation": "elu",  # Activation function
        "context_upsample_factor": 4,  # Upsample factor (512x -> 128x = 4x)
    },
    "generator_32": {
        "source_vqvae": "vqvae_128",  # Context from 128x codes
        "dest_vqvae": "vqvae_32",  # Generates codes for 32x compression
        "lstm_units": 256,
        "lstm_layers": 2,
        # Context model configuration
        "context_dim": 256,  # Output dimension of context features
        "context_channels": 256,  # Intermediate channels in context model dilated CNN
        "context_dilations": [1, 2, 4, 8, 16, 32],  # Dilation rates for each layer
        "context_kernel_size": 3,  # Kernel size for dilated conv layers
        "context_activation": "elu",  # Activation function
        "context_upsample_factor": 4,  # Upsample factor (128x -> 32x = 4x)
    },
    "generator_8": {
        "source_vqvae": "vqvae_32",  # Context from 32x codes
        "dest_vqvae": "vqvae_8",  # Generates codes for 8x compression
        "lstm_units": 256,
        "lstm_layers": 2,
        # Context model configuration
        "context_dim": 256,  # Output dimension of context features
        "context_channels": 256,  # Intermediate channels in context model dilated CNN
        "context_dilations": [1, 2, 4, 8, 16, 32],  # Dilation rates for each layer
        "context_kernel_size": 3,  # Kernel size for dilated conv layers
        "context_activation": "elu",  # Activation function
        "context_upsample_factor": 4,  # Upsample factor (32x -> 8x = 4x)
    },
}
