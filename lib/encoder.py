"""
VQ-VAE Encoder, Decoder, and Codebook classes.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from lib.layers import Codebook
from lib.config import ENCODER_CONFIGS


class Encoder(keras.Model):
    """
    VQ-VAE Encoder that converts audio to latent representations.
    """
    
    def __init__(self, config, **kwargs):
        """
        Initialize Encoder with configuration.
        
        Args:
            config: Dictionary containing encoder configuration:
                - compression_rate: Compression factor
                - encoder_layers: List of encoder layer configs
                - code_dim: Dimension of codebook vectors
        """
        super().__init__(**kwargs)
        self.config = config
        self.compression_rate = config["compression_rate"]
        self.code_dim = config["code_dim"]
        
        # Verify compression rate matches architecture
        actual_compression = 1
        for layer in config["encoder_layers"]:
            stride = layer.get("stride", 1)
            actual_compression *= stride
        assert actual_compression == self.compression_rate, \
            f"Compression rate mismatch: config says {self.compression_rate}x but architecture has {actual_compression}x compression (product of encoder strides)"
        
        # Build layers
        self.reshape = layers.Reshape((-1, 1))
        self.conv_layers = []
        for layer_cfg in config["encoder_layers"]:
            channels = layer_cfg["channels"]
            kernel = layer_cfg["kernel"]
            stride = layer_cfg["stride"]
            activation = layer_cfg.get("activation", "elu")
            
            self.conv_layers.append(
                layers.Conv1D(
                    channels, kernel, 
                    strides=stride, 
                    padding='same', 
                    activation=activation
                )
            )
        
        # Final bottleneck layer
        self.bottleneck = layers.Conv1D(
            self.code_dim, 1, 
            dtype='float32', 
            name='encoder_output'
        )
    
    def call(self, inputs, training=False):
        """
        Forward pass: encode audio to latent representations.
        
        Args:
            inputs: Audio tensor of shape (batch, seq_len)
            training: Whether in training mode
            
        Returns:
            latents: Latent tensor of shape (batch, code_seq_len, code_dim)
        """
        x = self.reshape(inputs)
        for conv_layer in self.conv_layers:
            x = conv_layer(x, training=training)
        x = self.bottleneck(x, training=training)
        return x
    
    def encode(self, audio, training=False):
        """
        Encode audio to latent representations.
        
        Args:
            audio: Audio tensor of shape (batch, seq_len)
            training: Whether in training mode
            
        Returns:
            latents: Latent tensor of shape (batch, code_seq_len, code_dim)
        """
        return self(audio, training=training)


class Decoder(keras.Model):
    """
    VQ-VAE Decoder that converts latent representations to audio.
    """
    
    def __init__(self, config, **kwargs):
        """
        Initialize Decoder with configuration.
        
        Args:
            config: Dictionary containing decoder configuration:
                - compression_rate: Compression factor (for verification)
                - decoder_layers: List of decoder layer configs
                - code_dim: Dimension of codebook vectors
        """
        super().__init__(**kwargs)
        self.config = config
        self.compression_rate = config["compression_rate"]
        self.code_dim = config["code_dim"]
        
        # Verify decoder expansion rate matches compression rate
        actual_expansion = 1
        for layer in config["decoder_layers"]:
            if layer.get("transpose", False):
                stride = layer.get("stride", 1)
                actual_expansion *= stride
        assert actual_expansion == self.compression_rate, \
            f"Decoder expansion mismatch: compression rate is {self.compression_rate}x but decoder expands by {actual_expansion}x (product of transpose conv strides)"
        
        # Build layers
        self.conv_layers = []
        for layer_cfg in config["decoder_layers"]:
            channels = layer_cfg["channels"]
            kernel = layer_cfg["kernel"]
            stride = layer_cfg["stride"]
            activation = layer_cfg.get("activation", "elu")
            is_transpose = layer_cfg.get("transpose", False)
            
            if is_transpose:
                self.conv_layers.append(
                    layers.Conv1DTranspose(
                        channels, kernel,
                        strides=stride,
                        padding='same',
                        activation=activation
                    )
                )
            else:
                self.conv_layers.append(
                    layers.Conv1D(
                        channels, kernel,
                        strides=stride,
                        padding='same',
                        activation=activation
                    )
                )
        
        # Final output layers (tanh * sigmoid)
        self.output_tanh = layers.Conv1D(1, 1, activation='tanh')
        self.output_sigmoid = layers.Conv1D(1, 1, activation='sigmoid')
        self.flatten = layers.Flatten(dtype='float32', name='decoder_output')
    
    def call(self, inputs, training=False):
        """
        Forward pass: decode code vectors to audio.
        
        Args:
            inputs: Code vectors tensor of shape (batch, code_seq_len, code_dim)
            training: Whether in training mode
            
        Returns:
            audio: Reconstructed audio tensor of shape (batch, audio_seq_len)
        """
        x = inputs
        for conv_layer in self.conv_layers:
            x = conv_layer(x, training=training)
        
        x_tanh = self.output_tanh(x, training=training)
        x_sigmoid = self.output_sigmoid(x, training=training)
        x = x_tanh * x_sigmoid
        x = self.flatten(x)
        return x
    
    def decode(self, code_vectors, training=False):
        """
        Decode code vectors to audio.
        
        Args:
            code_vectors: Code vectors tensor of shape (batch, code_seq_len, code_dim)
            training: Whether in training mode
            
        Returns:
            audio: Reconstructed audio tensor of shape (batch, audio_seq_len)
        """
        return self(code_vectors, training=training)


class CodebookManager(keras.Model):
    """
    Manages the VQ-VAE codebook for quantization.
    """
    
    def __init__(self, config, **kwargs):
        """
        Initialize CodebookManager with configuration.
        
        Args:
            config: Dictionary containing codebook configuration:
                - num_codes: Codebook size
                - code_dim: Dimension of codebook vectors
        """
        super().__init__(**kwargs)
        self.config = config
        self.num_codes = config["num_codes"]
        self.code_dim = config["code_dim"]
        
        self.codebook_layer = Codebook(
            self.num_codes,
            codes_initializer='random_normal',
            dtype='float32',
            name='codebook'
        )
    
    def call(self, inputs, training=False):
        """
        Forward pass: quantize latents using the codebook.
        
        Args:
            inputs: Latent tensor of shape (batch, seq_len, code_dim)
            training: Whether in training mode
            
        Returns:
            quantized: Quantized vectors of shape (batch, seq_len, code_dim)
            codes: Integer codes of shape (batch, seq_len)
        """
        quantized, codes = self.codebook_layer(inputs, training=training)
        return quantized, codes
    
    def quantize(self, latents, training=False):
        """
        Quantize latents using the codebook.
        
        Args:
            latents: Latent tensor of shape (batch, seq_len, code_dim)
            training: Whether in training mode
            
        Returns:
            quantized: Quantized vectors of shape (batch, seq_len, code_dim)
            codes: Integer codes of shape (batch, seq_len)
        """
        return self(latents, training=training)
    
    def gather(self, codes):
        """
        Gather codebook vectors from integer codes.
        
        Args:
            codes: Integer codes tensor of shape (batch, seq_len)
            
        Returns:
            code_vectors: Codebook vectors of shape (batch, seq_len, code_dim)
        """
        return self.codebook_layer.gather(codes)
