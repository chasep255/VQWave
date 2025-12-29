"""
Global configuration constants for audio processing.
"""

# Audio sample rate (Hz)
SAMPLE_RATE = 22050

# VQ-VAE configuration presets
ENCODER_CONFIGS = {
    "vqvae_512": {
        "compression_rate": 512,
        "num_codes": 4096,
        "code_dim": 64,
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
}

