"""
Generator classes for autoregressive code prediction.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input, Model

from vqwave.config import ENCODER_CONFIGS, GENERATOR_CONFIGS


class ContextModel(Model):
    """
    Context model for conditioning generators on lower-resolution codes.
    
    Processes lower-res codes (e.g., 512x) with dilated CNN and upsamples
    to condition higher-res generation (e.g., 128x).
    """
    
    def __init__(self, num_codes, embedding_dim, context_layers, 
                 name='context_model', **kwargs):
        """
        Initialize ContextModel with layer-based configuration.
        
        Args:
            num_codes: Size of codebook for lower-res codes
            embedding_dim: Dimension of code embeddings
            context_layers: List of layer configs, each with:
                - channels: Number of output channels
                - kernel: Kernel size
                - stride: Stride (for transpose conv, this is the upsample factor)
                - dilation: Dilation rate (for regular conv)
                - activation: Activation function
                - transpose: If True, use Conv1DTranspose (for upsampling)
        """
        
        # Define input (variable length sequence of lower-res integer codes)
        input_codes = Input((None,), dtype='int32', name='context_codes_input')
        
        # Embed integer codes
        x = layers.Embedding(num_codes, embedding_dim, name='context_embedding')(input_codes)
        
        # Process layers
        for i, layer_config in enumerate(context_layers):
            channels = layer_config["channels"]
            kernel = layer_config["kernel"]
            activation = layer_config["activation"]
            is_transpose = layer_config.get("transpose", False)
            
            if is_transpose:
                # Transpose convolution for upsampling
                stride = layer_config["stride"]
                x = layers.Conv1DTranspose(
                    channels,
                    kernel,
                    strides=stride,
                    padding='same',
                    activation=activation,
                    dtype='float32',
                    name=f'context_conv{i+1}'
                )(x)
            else:
                # Regular dilated convolution
                dilation = layer_config.get("dilation", 1)
                x = layers.Conv1D(
                    channels,
                    kernel,
                    padding='same',
                    dilation_rate=dilation,
                    activation=activation,
                    name=f'context_conv{i+1}'
                )(x)
        
        super().__init__(inputs=input_codes, outputs=x, name=name, **kwargs)
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        self.context_layers = context_layers


class Generator(Model):
    """
    First-level generator: 2-stacked LSTM for autoregressive code prediction.
    
    Takes integer codes from the VQ-VAE codebook and predicts the next code.
    Can optionally be conditioned on context from lower-resolution codes.
    Supports both training (non-stateful) and inference (stateful) modes.
    """
    
    def __init__(self, num_codes, embedding_dim=32, lstm_units=1024, lstm_layers=2, 
                 context_dim=None, stateful=False, batch_size=None, name='generator', **kwargs):
        """
        Initialize Generator with configuration.
        
        Args:
            num_codes: Size of codebook (vocab size for prediction)
            embedding_dim: Dimension of code embeddings
            lstm_units: Number of units in each LSTM layer
            lstm_layers: Number of stacked LSTM layers
            context_dim: Dimension of context features (if None, no context conditioning)
            stateful: If True, LSTM layers are stateful (for inference mode)
            batch_size: Batch size (required if stateful=True)
        """
        if stateful and batch_size is None:
            raise ValueError("batch_size must be specified when stateful=True")
        
        # Define input (variable length sequence of integer codes)
        # For stateful mode, batch_size must be fixed
        input_codes = Input(
            (None,), 
            dtype='int32', 
            batch_size=batch_size if stateful else None,
            name='codes_input'
        )
        
        # Optional context input (from ContextModel)
        if context_dim is not None:
            input_context = Input(
                (None, context_dim), 
                batch_size=batch_size if stateful else None,
                name='context_input'
            )
            inputs = [input_codes, input_context]
        else:
            input_context = None
            inputs = input_codes
        
        # Embed integer codes
        x = layers.Embedding(num_codes, embedding_dim, name='code_embedding')(input_codes)
        
        # Concatenate context to input embedding if provided
        if context_dim is not None:
            # Concatenate along channel dimension: [batch, time, embedding_dim + context_dim]
            x = layers.Concatenate(axis=-1, name='concat_context_input')([x, input_context])
        
        # Stacked LSTM layers with context concatenation (stateful for inference mode)
        lstm_layers_list = []
        for i in range(lstm_layers):
            # LSTM layer
            lstm_layer = layers.LSTM(
                lstm_units,
                return_sequences=True,
                stateful=stateful,
                name=f'lstm_{i+1}'
            )
            lstm_layers_list.append(lstm_layer)
            x = lstm_layer(x)

        if context_dim is not None:
            x = layers.Concatenate(axis=-1, name='concat_context_output')([x, input_context])

        x = layers.Conv1D(
            lstm_units, 1,
            activation='swish',
            name='output_conv'
        )(x)
        
        # Output logits over codebook (one logit per code)
        outputs = layers.Conv1D(
            num_codes, 1,
            dtype='float32',
            name='output_logits'
        )(x)
        
        super().__init__(inputs=inputs, outputs=outputs, name=name, **kwargs)
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.lstm_layers = lstm_layers
        self.context_dim = context_dim
        self.stateful = stateful
        self.batch_size = batch_size
        self.lstm_layers_list = lstm_layers_list
    
    def reset_states(self):
        """
        Reset LSTM states (for stateful inference mode).
        Call this before starting a new sequence.
        """
        if not self.stateful:
            raise RuntimeError("reset_states() can only be called when stateful=True")
        for lstm_layer in self.lstm_layers_list:
            lstm_layer.reset_states()
    
    def get_states(self):
        """
        Get current LSTM states (for stateful inference mode).
        Returns tuple of states, one per LSTM layer, where each state is (h, c).
        """
        if not self.stateful:
            raise RuntimeError("get_states() can only be called when stateful=True")
        return tuple(lstm_layer.states for lstm_layer in self.lstm_layers_list)
    
    def set_states(self, states):
        """
        Set LSTM states (for stateful inference mode).
        
        Args:
            states: Tuple of states, one per LSTM layer, where each state is (h, c).
        """
        if not self.stateful:
            raise RuntimeError("set_states() can only be called when stateful=True")
        if len(states) != len(self.lstm_layers_list):
            raise ValueError(f"Expected {len(self.lstm_layers_list)} states, got {len(states)}")
        for lstm_layer, state in zip(self.lstm_layers_list, states):
            lstm_layer.states = state


def create_generator(generator_config, stateful=False, batch_size=None, name=None):
    """
    Create a Generator (and ContextModel if needed) from a config.
    
    Args:
        generator_config: Either a string key from GENERATOR_CONFIGS or a dict with generator config
        stateful: If True, create stateful generator for inference mode
        batch_size: Batch size (required if stateful=True)
        name: Optional name prefix for models (defaults to generator config key)
    
    Returns:
        If context is needed: (generator, context_model)
        If unconditional: generator
    """
    # Get config dict
    if isinstance(generator_config, str):
        if generator_config not in GENERATOR_CONFIGS:
            raise ValueError(f"Unknown generator config: {generator_config}")
        config = GENERATOR_CONFIGS[generator_config]
        config_name = generator_config
    else:
        config = generator_config
        config_name = name or "generator"
    
    # Get destination VQ-VAE config (target codes we're generating)
    dest_vqvae_key = config["dest_vqvae"]
    if dest_vqvae_key not in ENCODER_CONFIGS:
        raise ValueError(f"Unknown VQ-VAE config: {dest_vqvae_key}")
    dest_vqvae = ENCODER_CONFIGS[dest_vqvae_key]
    
    # Derive generator parameters from dest VQ-VAE
    num_codes = dest_vqvae["num_codes"]
    embedding_dim = dest_vqvae["code_dim"]
    lstm_units = config["lstm_units"]
    lstm_layers = config.get("lstm_layers", 2)
    
    # Check if we need context
    source_vqvae_key = config.get("source_vqvae")
    context_model = None
    
    if source_vqvae_key is not None:
        # Get source VQ-VAE config (context codes)
        if source_vqvae_key not in ENCODER_CONFIGS:
            raise ValueError(f"Unknown VQ-VAE config: {source_vqvae_key}")
        source_vqvae = ENCODER_CONFIGS[source_vqvae_key]
        
        # Derive context model parameters from source VQ-VAE
        context_num_codes = source_vqvae["num_codes"]
        context_embedding_dim = source_vqvae["code_dim"]
        
        # Get layer-based context configuration (new format)
        context_layers = config.get("context_layers")
        
        # Fallback to old format for backward compatibility
        if context_layers is None:
            context_dim = config.get("context_dim", 512)
            context_channels = config.get("context_channels", 512)
            context_dilations = config.get("context_dilations", [1, 2, 4, 8, 16, 32])
            context_kernel_size = config.get("context_kernel_size", 3)
            context_activation = config.get("context_activation", "elu")
            context_upsample_factor = config.get("context_upsample_factor", 4)
            
            # Convert old format to new format
            context_layers = [
                {
                    "channels": context_channels,
                    "kernel": context_kernel_size,
                    "dilation": dilation,
                    "activation": context_activation,
                }
                for dilation in context_dilations
            ]
            # Add transpose layer for upsampling
            context_layers.append({
                "channels": context_dim,
                "kernel": context_upsample_factor,
                "stride": context_upsample_factor,
                "activation": context_activation,
                "transpose": True,
            })
        
        # Extract context_dim from last layer for generator
        context_dim = context_layers[-1]["channels"]
        
        # Create context model
        context_model = ContextModel(
            num_codes=context_num_codes,
            embedding_dim=context_embedding_dim,
            context_layers=context_layers,
            name=f"{config_name}_context"
        )
        
        # Generator with context
        generator = Generator(
            num_codes=num_codes,
            embedding_dim=embedding_dim,
            lstm_units=lstm_units,
            lstm_layers=lstm_layers,
            context_dim=context_dim,
            stateful=stateful,
            batch_size=batch_size,
            name=config_name
        )
        
        return generator, context_model
    else:
        # Unconditional generator
        generator = Generator(
            num_codes=num_codes,
            embedding_dim=embedding_dim,
            lstm_units=lstm_units,
            lstm_layers=lstm_layers,
            context_dim=None,
            stateful=stateful,
            batch_size=batch_size,
            name=config_name
        )
        
        return generator, None

