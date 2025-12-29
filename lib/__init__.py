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
from .config import ENCODER_CONFIGS
from .generator import create_generator
from .critic import create_critic
from .rnn_decoder import create_rnn_decoder

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
    # Generator/Critic
    'create_generator',
    'create_critic',
    'create_rnn_decoder',
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

