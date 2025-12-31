"""
Generator classes for autoregressive code prediction.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input, Model

from lib.config import ENCODER_CONFIGS, GENERATOR_CONFIGS


class ContextModel(Model):
    """
    Context model for conditioning generators on lower-resolution codes.
    
    Processes lower-res codes (e.g., 512x) with dilated CNN and upsamples 4x
    to condition higher-res generation (e.g., 128x).
    """
    
    def __init__(self, num_codes, embedding_dim=64, context_dim=512, context_channels=512, name='context_model', **kwargs):
        """
        Initialize ContextModel with configuration.
        
        Args:
            num_codes: Size of codebook for lower-res codes
            embedding_dim: Dimension of code embeddings
            context_dim: Dimension of output context features
            context_channels: Number of channels in intermediate dilated CNN layers
        """
        # Define input (variable length sequence of lower-res integer codes)
        input_codes = Input((None,), dtype='int32', name='context_codes_input')
        
        # Embed integer codes
        x = layers.Embedding(num_codes, embedding_dim, name='context_embedding')(input_codes)
        
        # Dilated CNN layers for large receptive field
        # Start with smaller dilation, increase for wider context
        x = layers.Conv1D(context_channels, 3, padding='same', activation='elu', name='context_conv1')(x)
        x = layers.Conv1D(context_channels, 3, padding='same', dilation_rate=2, activation='elu', name='context_conv2')(x)
        x = layers.Conv1D(context_channels, 3, padding='same', dilation_rate=4, activation='elu', name='context_conv3')(x)
        x = layers.Conv1D(context_channels, 3, padding='same', dilation_rate=8, activation='elu', name='context_conv4')(x)
        x = layers.Conv1D(context_channels, 3, padding='same', dilation_rate=16, activation='elu', name='context_conv5')(x)
        x = layers.Conv1D(context_channels, 3, padding='same', dilation_rate=32, activation='elu', name='context_conv6')(x)

        
        # Upsample 4x with transpose convolution
        outputs = layers.Conv1DTranspose(
            context_dim, 4,
            strides=4,
            padding='same',
            activation='elu',
            dtype='float32',
            name='context_output'
        )(x)
        
        super().__init__(inputs=input_codes, outputs=outputs, name=name, **kwargs)
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        self.context_dim = context_dim
        self.context_channels = context_channels


class Generator(Model):
    """
    First-level generator: 2-stacked LSTM for autoregressive code prediction.
    
    Takes integer codes from the VQ-VAE codebook and predicts the next code.
    Can optionally be conditioned on context from lower-resolution codes.
    Supports both training (non-stateful) and inference (stateful) modes.
    """
    
    def __init__(self, num_codes, embedding_dim=64, lstm_units=1024, context_dim=None, 
                 stateful=False, batch_size=None, name='generator', **kwargs):
        """
        Initialize Generator with configuration.
        
        Args:
            num_codes: Size of codebook (vocab size for prediction)
            embedding_dim: Dimension of code embeddings
            lstm_units: Number of units in each LSTM layer
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
        
        # Add context if provided (concatenate or add - using add for simplicity)
        if context_dim is not None:
            # Project context to match embedding dim if needed
            if context_dim != embedding_dim:
                context_proj = layers.Dense(embedding_dim, use_bias=False, name='context_projection')(input_context)
            else:
                context_proj = input_context
            # Add context to embeddings
            # Note: ContextModel upsamples 4x, so context sequence length should match codes length
            x = x + context_proj
        
        # First LSTM layer (stateful for inference mode)
        lstm1 = layers.LSTM(
            lstm_units,
            return_sequences=True,
            stateful=stateful,
            name='lstm_1'
        )
        x = lstm1(x)
        
        # Second LSTM layer (stateful for inference mode)
        lstm2 = layers.LSTM(
            lstm_units,
            return_sequences=True,
            stateful=stateful,
            name='lstm_2'
        )
        x = lstm2(x)
        
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
        self.context_dim = context_dim
        self.stateful = stateful
        self.batch_size = batch_size
        self.lstm1 = lstm1
        self.lstm2 = lstm2
    
    def reset_states(self):
        """
        Reset LSTM states (for stateful inference mode).
        Call this before starting a new sequence.
        """
        if not self.stateful:
            raise RuntimeError("reset_states() can only be called when stateful=True")
        self.lstm1.reset_states()
        self.lstm2.reset_states()
    
    def get_states(self):
        """
        Get current LSTM states (for stateful inference mode).
        Returns tuple of (lstm1_state, lstm2_state) where each state is (h, c).
        """
        if not self.stateful:
            raise RuntimeError("get_states() can only be called when stateful=True")
        return (self.lstm1.states, self.lstm2.states)
    
    def set_states(self, states):
        """
        Set LSTM states (for stateful inference mode).
        
        Args:
            states: Tuple of (lstm1_state, lstm2_state) where each state is (h, c).
        """
        if not self.stateful:
            raise RuntimeError("set_states() can only be called when stateful=True")
        lstm1_state, lstm2_state = states
        self.lstm1.states = lstm1_state
        self.lstm2.states = lstm2_state


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
        context_dim = config.get("context_dim", 512)
        context_channels = config.get("context_channels", 512)
        
        # Create context model
        context_model = ContextModel(
            num_codes=context_num_codes,
            embedding_dim=context_embedding_dim,
            context_dim=context_dim,
            context_channels=context_channels,
            name=f"{config_name}_context"
        )
        
        # Generator with context
        generator = Generator(
            num_codes=num_codes,
            embedding_dim=embedding_dim,
            lstm_units=lstm_units,
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
            context_dim=None,
            stateful=stateful,
            batch_size=batch_size,
            name=config_name
        )
        
        return generator, None

