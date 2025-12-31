"""
VQ-VAE Encoder, Decoder, and Codebook classes.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input, Model

from lib.layers import Codebook
from lib.config import ENCODER_CONFIGS


class Encoder(Model):
    """
    VQ-VAE Encoder that converts audio to latent representations.
    """
    
    def __init__(self, config, name='encoder', **kwargs):
        """
        Initialize Encoder with configuration.
        
        Args:
            config: Dictionary containing encoder configuration:
                - compression_rate: Compression factor
                - encoder_layers: List of encoder layer configs
                - code_dim: Dimension of codebook vectors
        """
        # Verify compression rate matches architecture
        actual_compression = 1
        for layer in config["encoder_layers"]:
            stride = layer.get("stride", 1)
            actual_compression *= stride
        assert actual_compression == config["compression_rate"], \
            f"Compression rate mismatch: config says {config['compression_rate']}x but architecture has {actual_compression}x compression (product of encoder strides)"
        
        # Define input (variable length sequence)
        input_audio = Input((None,), name='audio_input')
        
        # Reshape to add channel dimension
        x = layers.Reshape((-1, 1))(input_audio)
        
        # Build encoder layers
        for layer_cfg in config["encoder_layers"]:
            channels = layer_cfg["channels"]
            kernel = layer_cfg["kernel"]
            stride = layer_cfg.get("stride", 1)
            activation = layer_cfg.get("activation", "elu")
            
            x = layers.Conv1D(
                channels, kernel, 
                strides=stride, 
                padding='same', 
                activation=activation
            )(x)
        
        # Final bottleneck layer
        outputs = layers.Conv1D(
            config["code_dim"], 1, 
            dtype='float32', 
            name='encoder_output'
        )(x)
        
        super().__init__(inputs=input_audio, outputs=outputs, name=name, **kwargs)
        self.config = config
        self.compression_rate = config["compression_rate"]
        self.code_dim = config["code_dim"]


class Decoder(Model):
    """
    VQ-VAE Decoder that converts latent representations to audio.
    """
    
    def __init__(self, config, name='decoder', **kwargs):
        """
        Initialize Decoder with configuration.
        
        Args:
            config: Dictionary containing decoder configuration:
                - compression_rate: Compression factor (for verification)
                - decoder_layers: List of decoder layer configs
                - code_dim: Dimension of codebook vectors
        """
        # Verify decoder expansion rate matches compression rate
        actual_expansion = 1
        for layer in config["decoder_layers"]:
            if layer.get("transpose", False):
                stride = layer.get("stride", 1)
                actual_expansion *= stride
        assert actual_expansion == config["compression_rate"], \
            f"Decoder expansion mismatch: compression rate is {config['compression_rate']}x but decoder expands by {actual_expansion}x (product of transpose conv strides)"
        
        # Define input (variable length sequence)
        input_codes = Input((None, config["code_dim"]), name='code_vectors_input')
        
        # Build decoder layers
        x = input_codes
        for layer_cfg in config["decoder_layers"]:
            channels = layer_cfg["channels"]
            kernel = layer_cfg["kernel"]
            stride = layer_cfg.get("stride", 1)
            activation = layer_cfg.get("activation", "elu")
            is_transpose = layer_cfg.get("transpose", False)
            
            if is_transpose:
                x = layers.Conv1DTranspose(
                    channels, kernel,
                    strides=stride,
                    padding='same',
                    activation=activation
                )(x)
            else:
                x = layers.Conv1D(
                    channels, kernel,
                    strides=stride,
                    padding='same',
                    activation=activation
                )(x)
        
        # Final output layers (tanh * sigmoid) - ensure float32 output
        x = layers.Conv1D(1, 1, activation='tanh')(x)
        outputs = layers.Flatten(dtype='float32', name='decoder_output')(x)
        
        super().__init__(inputs=input_codes, outputs=outputs, name=name, **kwargs)
        self.config = config
        self.compression_rate = config["compression_rate"]
        self.code_dim = config["code_dim"]


class CodebookManager(Model):
    """
    Manages the VQ-VAE codebook for quantization.
    """
    
    def __init__(self, config, name='codebook_manager', **kwargs):
        """
        Initialize CodebookManager with configuration.
        
        Args:
            config: Dictionary containing codebook configuration:
                - num_codes: Codebook size
                - code_dim: Dimension of codebook vectors
        """
        # Define input (variable length sequence)
        input_latents = Input((None, config["code_dim"]), name='latents_input')
        
        # Codebook layer
        codebook_layer = Codebook(
            config["num_codes"],
            codes_initializer='random_normal',
            dtype='float32',
            name='codebook'
        )
        
        # Quantize
        quantized, codes = codebook_layer(input_latents)
        
        # Create model with multiple outputs
        super().__init__(inputs=input_latents, outputs=[quantized, codes], name=name, **kwargs)
        self.config = config
        self.num_codes = config["num_codes"]
        self.code_dim = config["code_dim"]
        self.codebook_layer = codebook_layer
    
    def gather(self, codes):
        """
        Gather codebook vectors from integer codes.
        
        Args:
            codes: Integer codes tensor of shape (batch, seq_len)
            
        Returns:
            code_vectors: Codebook vectors of shape (batch, seq_len, code_dim)
        """
        return self.codebook_layer.gather(codes)
