"""
Global configuration constants for audio processing.
"""

# Audio sample rate (Hz)
SAMPLE_RATE = 22050

# VQ-VAE configuration presets
#
# Two compression presets are provided (vqvae_256 and vqvae_512), trained with a
# reconstruction loss plus an optional WGAN-GP adversarial loss from the `critic`
# stack below (see scripts/train_vqvae.py and vqwave/critic.py). The critic is
# training-only and pushes the decoder toward realistic audio; the reconstruction
# loss remains the anchor tying the codes to the input.
#
# Encoder/decoder layers take an optional `dilation` (default 1) to widen the
# receptive field without extra parameters. Only valid on stride-1, non-transpose
# convs -- i.e. the frame-rate refinement stacks.
ENCODER_CONFIGS = {
    "vqvae_256": {
        "compression_rate": 256,
        "num_codes": 4096,
        "code_dim": 16,
        "encoder_layers": [
            # 4^4 = 256
            {"channels": 32, "kernel": 12, "stride": 4, "activation": "elu"},
            {"channels": 64, "kernel": 12, "stride": 4, "activation": "elu"},
            {"channels": 128, "kernel": 12, "stride": 4, "activation": "elu"},
            {"channels": 256, "kernel": 12, "stride": 4, "activation": "elu"},
            # Frame-rate refinement. Dilations ramp 1,2,4 then two stride-1 cleanup
            # layers: at kernel 9 every tap spacing stays inside the previous layer's
            # receptive field, so coverage is gapless. Encoder RF = 19368 (0.88s).
            {"channels": 512, "kernel": 9, "stride": 1, "dilation": 1, "activation": "elu"},
            {"channels": 512, "kernel": 9, "stride": 1, "dilation": 2, "activation": "elu"},
            {"channels": 512, "kernel": 9, "stride": 1, "dilation": 4, "activation": "elu"},
            {"channels": 512, "kernel": 9, "stride": 1, "dilation": 1, "activation": "elu"},
            {"channels": 512, "kernel": 9, "stride": 1, "dilation": 1, "activation": "elu"},

        ],
        "decoder_layers": [
            # Mirror refinement first (same dilation order as the encoder).
            {"channels": 512, "kernel": 9, "stride": 1, "dilation": 1, "activation": "elu"},
            {"channels": 512, "kernel": 9, "stride": 1, "dilation": 2, "activation": "elu"},
            {"channels": 512, "kernel": 9, "stride": 1, "dilation": 4, "activation": "elu"},
            {"channels": 512, "kernel": 9, "stride": 1, "dilation": 1, "activation": "elu"},
            {"channels": 512, "kernel": 9, "stride": 1, "dilation": 1, "activation": "elu"},

            # 4^4 = 256
            {"channels": 256, "kernel": 12, "stride": 4, "activation": "elu", "transpose": True},
            {"channels": 128, "kernel": 12, "stride": 4, "activation": "elu", "transpose": True},
            {"channels": 64, "kernel": 12, "stride": 4, "activation": "elu", "transpose": True},
            {"channels": 32, "kernel": 12, "stride": 4, "activation": "elu", "transpose": True},
            # Final projection to a single-channel waveform (Flatten -> [batch, time]).
            {"channels": 1, "kernel": 9, "stride": 1, "activation": "tanh"},
        ],
        # WGAN-GP critic (training only; see vqwave/critic.py). Scores a waveform
        # for realism -- strided convs downsample to per-window scores that are
        # averaged into one unbounded score per example. Deliberately has NO
        # normalization layers: BatchNorm would couple examples in a batch and
        # invalidate the per-example gradient penalty.
        #
        # Keep the total stride (4^6 = 4096) and the receptive field (55976 samples,
        # 2.5s) well under the training crop, or the averaged windows collapse into a
        # handful of near-identical scores over mostly-padding and the patch ensemble
        # stops carrying signal. The last layer is stride 1: it buys depth at the
        # widest channel count without costing windows.
        "critic": {
            "layers": [
                {"channels": 32,  "kernel": 12, "stride": 4, "alpha": 0.2},
                {"channels": 64,  "kernel": 12, "stride": 4, "alpha": 0.2},
                {"channels": 128, "kernel": 12, "stride": 4, "alpha": 0.2},
                {"channels": 256, "kernel": 12, "stride": 4, "alpha": 0.2},
                {"channels": 512, "kernel": 12, "stride": 4, "alpha": 0.2},
                {"channels": 1024, "kernel": 12, "stride": 4, "alpha": 0.2},
                {"channels": 1024, "kernel": 9, "stride": 1, "alpha": 0.2},
            ],
        },
    },
    "vqvae_512": {
        "compression_rate": 512,
        "num_codes": 2048,
        "code_dim": 16,
        "encoder_layers": [
            # 2 * 4^4 = 512
            {"channels": 32, "kernel": 6, "stride": 2, "activation": "elu"},
            {"channels": 64, "kernel": 12, "stride": 4, "activation": "elu"},
            {"channels": 128, "kernel": 12, "stride": 4, "activation": "elu"},
            {"channels": 256, "kernel": 12, "stride": 4, "activation": "elu"},
            {"channels": 512, "kernel": 12, "stride": 4, "activation": "elu"},
            # Frame-rate refinement. Same recipe as vqvae_256: dilations ramp 1,2,4
            # then two stride-1 cleanup layers. Gapless at kernel 9. Encoder RF =
            # 38740 samples (1.76s).
            {"channels": 512, "kernel": 9, "stride": 1, "dilation": 1, "activation": "elu"},
            {"channels": 512, "kernel": 9, "stride": 1, "dilation": 2, "activation": "elu"},
            {"channels": 512, "kernel": 9, "stride": 1, "dilation": 4, "activation": "elu"},
            {"channels": 512, "kernel": 9, "stride": 1, "dilation": 1, "activation": "elu"},
            {"channels": 512, "kernel": 9, "stride": 1, "dilation": 1, "activation": "elu"},
        ],
        "decoder_layers": [
            # Mirror refinement first (same dilation order as the encoder).
            {"channels": 512, "kernel": 9, "stride": 1, "dilation": 1, "activation": "elu"},
            {"channels": 512, "kernel": 9, "stride": 1, "dilation": 2, "activation": "elu"},
            {"channels": 512, "kernel": 9, "stride": 1, "dilation": 4, "activation": "elu"},
            {"channels": 512, "kernel": 9, "stride": 1, "dilation": 1, "activation": "elu"},
            {"channels": 512, "kernel": 9, "stride": 1, "dilation": 1, "activation": "elu"},

            # 2 * 4^4 = 512
            {"channels": 256, "kernel": 12, "stride": 4, "activation": "elu", "transpose": True},
            {"channels": 128, "kernel": 12, "stride": 4, "activation": "elu", "transpose": True},
            {"channels": 64, "kernel": 12, "stride": 4, "activation": "elu", "transpose": True},
            {"channels": 32, "kernel": 12, "stride": 4, "activation": "elu", "transpose": True},
            {"channels": 32, "kernel": 6, "stride": 2, "activation": "elu", "transpose": True},
            # Final projection to a single-channel waveform (Flatten -> [batch, time]).
            {"channels": 1, "kernel": 9, "stride": 1, "activation": "tanh"},
        ],
        # WGAN-GP critic (training only; see vqwave/critic.py). Scores a waveform
        # for realism -- strided convs downsample to per-window scores that are
        # averaged into one unbounded score per example. Deliberately has NO
        # normalization layers: BatchNorm would couple examples in a batch and
        # invalidate the per-example gradient penalty.
        #
        # Keep the total stride (4^6 = 4096) and the receptive field (55976 samples,
        # 2.5s) well under the training crop, or the averaged windows collapse into a
        # handful of near-identical scores over mostly-padding and the patch ensemble
        # stops carrying signal. The last layer is stride 1: it buys depth at the
        # widest channel count without costing windows.
        "critic": {
            "layers": [
                {"channels": 32,  "kernel": 12, "stride": 4, "alpha": 0.2},
                {"channels": 64,  "kernel": 12, "stride": 4, "alpha": 0.2},
                {"channels": 128, "kernel": 12, "stride": 4, "alpha": 0.2},
                {"channels": 256, "kernel": 12, "stride": 4, "alpha": 0.2},
                {"channels": 512, "kernel": 12, "stride": 4, "alpha": 0.2},
                {"channels": 1024, "kernel": 12, "stride": 4, "alpha": 0.2},
                {"channels": 1024, "kernel": 9, "stride": 1, "alpha": 0.2},
            ],
        },
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
