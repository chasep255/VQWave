"""
Generator classes for autoregressive code prediction.

A single unconditional generator predicts the VQ-VAE codes directly. Two
interchangeable architectures are provided:
  - TransformerGenerator: causal Transformer with a fixed context window.
  - Generator:            causal-conv / RNN stack (supports stateful inference).
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input, Model

from vqwave.config import ENCODER_CONFIGS, GENERATOR_CONFIGS
from vqwave.layers import ShiftBuffer, CausalQueue, CausalConv1D, PositionalEmbedding


class TransformerGenerator(Model):
    """
    Causal Transformer generator for autoregressive code prediction.

    Uses a fixed context window with learned position embeddings.
    """

    def __init__(self, num_codes, max_seq_len=512, embedding_dim=512,
                 num_layers=8, num_heads=8, key_dim=64, ff_dim=2048,
                 name='transformer_generator', **kwargs):
        """
        Initialize TransformerGenerator as a Functional Model.

        Args:
            num_codes: Size of codebook (vocab size for prediction)
            max_seq_len: Maximum sequence length (context window size)
            embedding_dim: Dimension of token and position embeddings (residual stream)
            num_layers: Number of transformer decoder layers
            num_heads: Number of attention heads
            key_dim: Dimension of each attention head (query/key/value per head)
            ff_dim: Hidden dimension of feed-forward network
        """
        # Define input (variable length sequence of integer codes)
        input_codes = Input((None,), dtype='int32', name='codes_input')

        # Token embeddings
        x = layers.Embedding(num_codes, embedding_dim, name='token_embedding')(input_codes)

        # Add position embeddings
        x = x + PositionalEmbedding(max_seq_len, embedding_dim, name='position_embedding')(x)

        # Transformer decoder layers (pre-norm architecture)
        for i in range(num_layers):
            # Self-attention with residual
            norm_x = layers.LayerNormalization(name=f'ln1_{i}')(x)
            mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, name=f'mha_{i}')
            attn_out = mha(query=norm_x, value=norm_x, key=norm_x, use_causal_mask=True)
            x = x + attn_out

            # Feed-forward with residual
            norm_x = layers.LayerNormalization(name=f'ln2_{i}')(x)
            ffn_out = layers.Dense(ff_dim, activation='gelu', name=f'ffn1_{i}')(norm_x)
            ffn_out = layers.Dense(embedding_dim, name=f'ffn2_{i}')(ffn_out)
            x = x + ffn_out

        # Final layer norm and output projection
        x = layers.LayerNormalization(name='final_ln')(x)
        x = layers.Dense(embedding_dim, activation='gelu', name='final_dense')(x)
        outputs = layers.Dense(num_codes, name='output_logits')(x)

        super().__init__(inputs=input_codes, outputs=outputs, name=name, **kwargs)

        # Store config
        self.num_codes = num_codes
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self._num_layers = num_layers
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.ff_dim = ff_dim

    def reset_states(self):
        """No-op for API compatibility with stateful generators."""
        pass


class Generator(Model):
    """
    Autoregressive generator for code prediction with configurable architecture.

    Takes integer codes from the VQ-VAE codebook and predicts the next code.
    Supports both training (non-stateful) and inference (stateful) modes.
    """

    def __init__(self, num_codes, embedding_dim=32, generator_layers=None,
                 stateful=False, batch_size=None, name='generator', **kwargs):
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

        # Embed integer codes
        x = layers.Embedding(num_codes, embedding_dim, name='code_embedding')(input_codes)

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

            else:
                raise ValueError(f"Unknown layer type: {layer_type}")

        # Output logits over codebook (one logit per code)
        outputs = layers.Conv1D(
            num_codes, 1,
            dtype='float32',
            name='output_logits'
        )(x)

        super().__init__(inputs=input_codes, outputs=outputs, name=name, **kwargs)
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        self.generator_layers = generator_layers
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
    Create a Generator from a config.

    Args:
        generator_config: Either a string key from GENERATOR_CONFIGS or a dict with generator config
        stateful: If True, create stateful generator for inference mode (ignored for transformer)
        batch_size: Batch size (required if stateful=True, ignored for transformer)
        name: Optional name prefix for the model (defaults to generator config key)

    Returns:
        The generator model (TransformerGenerator or Generator).
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

    # Check generator type (default to "rnn")
    generator_type = config.get("type", "rnn")

    if generator_type == "transformer":
        # Transformer generator (uses fixed-window inference)
        transformer_config = config.get("transformer", {})

        return TransformerGenerator(
            num_codes=num_codes,
            max_seq_len=transformer_config.get("max_seq_len", 512),
            embedding_dim=transformer_config.get("embedding_dim", 512),
            num_layers=transformer_config.get("num_layers", 8),
            num_heads=transformer_config.get("num_heads", 8),
            key_dim=transformer_config.get("key_dim", 64),
            ff_dim=transformer_config.get("ff_dim", 2048),
            name=config_name
        )

    # Standard RNN/Conv generator
    embedding_dim = dest_vqvae["code_dim"]

    generator_layers = config.get("generator_layers")
    if generator_layers is None:
        raise ValueError("generator_layers must be specified in generator config")

    return Generator(
        num_codes=num_codes,
        embedding_dim=embedding_dim,
        generator_layers=generator_layers,
        stateful=stateful,
        batch_size=batch_size,
        name=config_name
    )
