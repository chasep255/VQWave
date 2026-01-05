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
    "vqvae_4": {
        "compression_rate": 4,
        "num_codes": 256,
        "code_dim": 16,
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
        "lstm_units": 512,
        "lstm_layers": 2,
    },
    "generator_64": {
        "source_vqvae": "vqvae_1024",  # Context from 1024x codes
        "dest_vqvae": "vqvae_64",  # Generates codes for 64x compression
        "lstm_units": 256,
        "lstm_layers": 2,
        # Context model configuration
        "context_dim": 256,  # Output dimension of context features
        "context_channels": 512,  # Intermediate channels in context model dilated CNN
        "context_dilations": [1, 2, 4, 8, 16],  # Dilation rates for each layer
        "context_kernel_size": 3,  # Kernel size for dilated conv layers
        "context_activation": "elu",  # Activation function
        "context_upsample_factor": 16,  # Upsample factor (1024x -> 128x = 8x)
    },
    "generator_16": {
        "source_vqvae": "vqvae_128",  # Context from 128x codes
        "dest_vqvae": "vqvae_16",  # Generates codes for 16x compression
        "lstm_units": 128,
        "lstm_layers": 2,
        # Context model configuration
        "context_dim": 128,  # Output dimension of context features
        "context_channels": 256,  # Intermediate channels in context model dilated CNN
        "context_dilations": [1, 2, 4, 8, 16, 32],  # Dilation rates for each layer
        "context_kernel_size": 3,  # Kernel size for dilated conv layers
        "context_activation": "elu",  # Activation function
        "context_upsample_factor": 8,  # Upsample factor (128x -> 16x = 8x)
    },
}
