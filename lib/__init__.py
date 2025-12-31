# VQVAE Audio Generator Library

from .layers import (
    CausalQueue,
    ShiftBuffer,
    PositionalEmbedding,
    LearnedGaussianNoise,
    PrintLayer,
    Resize1D,
    TiedEmbedding,
    Codebook,
)
from .encoder import Encoder, Decoder, CodebookManager
from .config import ENCODER_CONFIGS, GENERATOR_CONFIGS
from .generator import Generator, ContextModel, create_generator

from .audio import (
    u16_to_f32,
    f32_to_u16,
    save_audio,
    load_audio,
    load_meta,
    mu_law,
    mu_law_inverse,
    mu_law_quantize,
    mu_law_dequantize,
    AudioDataset,
    AudioMix,
)

from .util import (
    LRWarmupWrapper,
    GradientAccumulator,
    CodebookRestarter,
)

__all__ = [
    # Layers
    'CausalQueue',
    'ShiftBuffer',
    'PositionalEmbedding',
    'LearnedGaussianNoise',
    'PrintLayer',
    'Resize1D',
    'TiedEmbedding',
    # Codebook
    'Codebook',
    # Encoder/Decoder
    'Encoder',
    'Decoder',
    'CodebookManager',
    'ENCODER_CONFIGS',
    'GENERATOR_CONFIGS',
    # Generator
    'Generator',
    'ContextModel',
    'create_generator',
    # Audio utilities
    'u16_to_f32',
    'f32_to_u16',
    'save_audio',
    'load_audio',
    'load_meta',
    'mu_law',
    'mu_law_inverse',
    'mu_law_quantize',
    'mu_law_dequantize',
    'AudioDataset',
    'AudioMix',
    # Training utilities
    'LRWarmupWrapper',
    'GradientAccumulator',
    'CodebookRestarter',
]

