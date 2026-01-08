"""
Generator classes for autoregressive code prediction.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input, Model

from vqwave.config import ENCODER_CONFIGS, GENERATOR_CONFIGS
from vqwave.layers import ShiftBuffer, CausalQueue, CausalConv1D


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
    Autoregressive generator for code prediction with configurable architecture.
    
    Takes integer codes from the VQ-VAE codebook and predicts the next code.
    Can optionally be conditioned on context from lower-resolution codes.
    Supports both training (non-stateful) and inference (stateful) modes.
    """
    
    def __init__(self, num_codes, embedding_dim=32, generator_layers=None,
                 context_dim=None, stateful=False, batch_size=None, name='generator', **kwargs):
        """
        Initialize Generator with layer-based configuration.
        
        Args:
            num_codes: Size of codebook (vocab size for prediction)
            embedding_dim: Dimension of code embeddings
            generator_layers: List of layer configs, each with:
                - {"type": "lstm", "units": 512} - LSTM layer
                - {"type": "gru", "units": 512} - GRU layer
                - {"type": "conv", "channels": 512, "kernel": 1, "activation": "elu"} - Conv1D layer
                - {"type": "causal_conv", "channels": 512, "kernel": 3, "dilation": 2, "activation": "elu"} - Causal Conv1D (stateful)
                - {"type": "highway", "channels": 512, "kernel": 3, "dilation": 2} - Highway layer (tanh transform, sigmoid gate)
                - {"type": "residual", "channels": 512, "kernel": 3, "dilation": 2, "activation": "elu"} - Pre-activated residual block
                - {"type": "layer_norm"} - LayerNormalization layer
                - {"type": "activation", "activation": "elu"} - Activation layer
                - {"type": "context_concat"} - Concatenate context at this point
            context_dim: Dimension of context features (if None, no context conditioning)
            stateful: If True, RNN layers are stateful (for inference mode)
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
        
        # Process configurable layers
        if generator_layers is None:
            raise ValueError("generator_layers must be provided explicitly in the generator config")
        
        layer_idx = 0
        
        for layer_config in generator_layers:
            layer_type = layer_config["type"]
            
            if layer_type == "lstm":
                units = layer_config["units"]
                rnn_layer = layers.LSTM(
                    units,
                    return_sequences=True,
                    stateful=stateful,
                    name=f'rnn_{layer_idx+1}'
                )
                x = rnn_layer(x)
                layer_idx += 1
                
            elif layer_type == "gru":
                units = layer_config["units"]
                rnn_layer = layers.GRU(
                    units,
                    return_sequences=True,
                    stateful=stateful,
                    name=f'rnn_{layer_idx+1}'
                )
                x = rnn_layer(x)
                layer_idx += 1
                
            elif layer_type == "conv":
                channels = layer_config["channels"]
                kernel = layer_config["kernel"]
                activation = layer_config.get("activation", "elu")
                x = layers.Conv1D(
                    channels,
                    kernel,
                    padding='same',  # Preserve sequence length
                    activation=activation,
                    name=f'conv_{layer_idx+1}'
                )(x)
                layer_idx += 1
                
            elif layer_type == "causal_conv":
                channels = layer_config["channels"]
                kernel = layer_config["kernel"]
                activation = layer_config.get("activation", "elu")
                dilation = layer_config.get("dilation", 1)
                
                causal_layer = CausalConv1D(
                    channels, kernel, dilation=dilation, activation=activation,
                    stateful=stateful, name=f'causal_conv_{layer_idx+1}'
                )
                x = causal_layer(x)
                layer_idx += 1
                
            elif layer_type == "layer_norm":
                x = layers.LayerNormalization(name=f'layer_norm_{layer_idx+1}')(x)
                layer_idx += 1
                
            elif layer_type == "activation":
                activation = layer_config.get("activation", "elu")
                x = layers.Activation(activation, name=f'activation_{layer_idx+1}')(x)
                layer_idx += 1
            
            elif layer_type == "highway":
                # Highway network: y = H(x) * T(x) + x * (1 - T(x))
                # H = transform (causal conv + tanh)
                # T = gate (causal conv + sigmoid)
                channels = layer_config["channels"]
                kernel = layer_config["kernel"]
                dilation = layer_config.get("dilation", 1)
                
                receptive_field = (kernel - 1) * dilation + 1
                

                
                if stateful:
                    # Stateful mode: use buffer
                    # For kernel=2: use CausalQueue - queue handles dilation, conv uses dilation=1
                    # Otherwise: use ShiftBuffer for full receptive field
                    if kernel == 2:
                        causal_queue = CausalQueue(dilation)
                        buffered = causal_queue(x)
                        conv_dilation = 1  # Queue handles dilation
                    else:
                        shift_buffer = ShiftBuffer(receptive_field)
                        buffered = shift_buffer(x)
                        conv_dilation = dilation
                    
                    # Transform path H(x) - always tanh
                    h = layers.Conv1D(
                        channels, kernel, padding='valid', dilation_rate=conv_dilation,
                        activation='tanh', name=f'highway_h_{layer_idx+1}'
                    )(buffered)
                    
                    # Gate path T(x) - always sigmoid, bias initialized to -1 (gate starts closed)
                    t = layers.Conv1D(
                        channels, kernel, padding='valid', dilation_rate=conv_dilation,
                        activation='sigmoid', 
                        bias_initializer=tf.keras.initializers.Constant(-1.0),
                        name=f'highway_t_{layer_idx+1}'
                    )(buffered)
                else:
                    # Training mode: causal padding
                    h = layers.Conv1D(
                        channels, kernel, padding='causal', dilation_rate=dilation,
                        activation='tanh', name=f'highway_h_{layer_idx+1}'
                    )(x)
                    
                    # Gate path T(x) - always sigmoid, bias initialized to -1 (gate starts closed)
                    t = layers.Conv1D(
                        channels, kernel, padding='causal', dilation_rate=dilation,
                        activation='sigmoid',
                        bias_initializer=tf.keras.initializers.Constant(-1.0),
                        name=f'highway_t_{layer_idx+1}'
                    )(x)
                
                # Highway combination: y = H * T + x * (1 - T)
                x = h * t + x * (1.0 - t)
                layer_idx += 1
            
            elif layer_type == "residual":
                # Pre-activated residual block: x → activation → conv1 → activation → conv2 → + x
                channels = layer_config["channels"]
                kernel = layer_config["kernel"]
                activation = layer_config.get("activation", "elu")
                dilation = layer_config.get("dilation", 1)
                
                # Store input for residual connection - conv2 projects back to this channel size
                residual = x
                # Get input channel size (for conv2 output)
                residual_channels = residual.shape[-1]
                if residual_channels is None:
                    raise ValueError(
                        "Residual block requires a known (static) channel dimension on its input so conv2 can project back. "
                        "Add a projection conv before the residual block so channels are statically known."
                    )
                
                # Pre-activation: apply activation before conv1
                x = layers.Activation(activation, name=f'residual_act1_{layer_idx+1}')(x)
                
                # First causal conv (can change channel size)
                conv1 = CausalConv1D(
                    channels, kernel, dilation=dilation, activation=None,
                    stateful=stateful, name=f'residual_conv1_{layer_idx+1}'
                )
                x = conv1(x)
                
                # Pre-activation before conv2
                x = layers.Activation(activation, name=f'residual_act2_{layer_idx+1}')(x)
                
                # Second conv: always 1x1, projects back to residual input channel size
                x = layers.Conv1D(
                    residual_channels, 1, padding='same', activation=None,
                    name=f'residual_conv2_{layer_idx+1}'
                )(x)
                
                # Residual connection
                x = x + residual
                layer_idx += 1
                
            elif layer_type == "context_concat":
                if context_dim is None:
                    raise ValueError("context_concat layer requires context_dim to be set")
                x = layers.Concatenate(axis=-1, name=f'context_concat_{layer_idx+1}')([x, input_context])
                layer_idx += 1
                
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
        
        # Output logits over codebook (one logit per code)
        outputs = layers.Conv1D(
            num_codes, 1,
            dtype='float32',
            name='output_logits'
        )(x)
        
        super().__init__(inputs=inputs, outputs=outputs, name=name, **kwargs)
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        self.generator_layers = generator_layers
        self.context_dim = context_dim
        self.stateful = stateful
        self.batch_size = batch_size
    
    def reset_states(self):
        """
        Reset RNN states (for stateful inference mode).
        Call this before starting a new sequence.
        """
        if not self.stateful:
            raise RuntimeError("reset_states() can only be called when stateful=True")
        # Find all layers with reset_states method (LSTM, GRU, CausalConv1D, CausalQueue, ShiftBuffer)
        for layer in self.layers:
            if hasattr(layer, 'reset_states'):
                layer.reset_states()


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
    
    # Get generator layers configuration
    generator_layers = config.get("generator_layers")
    if generator_layers is None:
        raise ValueError("generator_layers must be specified in generator config")
    
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
        
        # Layer-based context configuration (required)
        context_layers = config.get("context_layers")
        if context_layers is None:
            raise ValueError(
                f"Generator config '{config_name}' uses context (source_vqvae={source_vqvae_key}) "
                "but is missing required 'context_layers' list."
            )
        
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
            generator_layers=generator_layers,
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
            generator_layers=generator_layers,
            context_dim=None,
            stateful=stateful,
            batch_size=batch_size,
            name=config_name
        )
        
        return generator, None

