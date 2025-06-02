import tensorflow as tf
from alpha_architecture.blocks.base_block import BaseBlock

class TransformerBlock(BaseBlock, tf.keras.layers.Layer):
    """
    A Transformer encoder block, combining Multi-Head Attention and Feed-Forward Network.
    This class inherits from both BaseBlock (for architecture generation integration)
    and tf.keras.layers.Layer (to function as a Keras layer).
    """
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, rate: float = 0.1, name: str = 'TransformerBlock', seed: int = None, **kwargs):
        """
        Initializes the TransformerBlock.

        Args:
            embed_dim (int): Dimensionality of the input and output embeddings.
            num_heads (int): Number of attention heads.
            ff_dim (int): Dimensionality of the inner layer of the feed-forward network.
            rate (float, optional): Dropout rate. Defaults to 0.1.
            name (str, optional): Name of the block/layer. Defaults to 'TransformerBlock'.
            seed (int, optional): Random seed for initializers and dropout. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        # Keras Layer initialization
        # We need to separate kwargs for tf.keras.layers.Layer and BaseBlock.
        # BaseBlock's __init__ will store all relevant args (embed_dim, num_heads, etc.)
        # in self.block_specific_config via its **kwargs.
        # tf.keras.layers.Layer.__init__ should only get its own valid kwargs.

        # BaseBlock expects 'name' and its specific params.
        # tf.keras.layers.Layer also expects 'name'.
        # Let's call Layer's init first, which sets self.name.
        # Then BaseBlock can use self.name.

        # Filter out kwargs meant for BaseBlock's primary parameters from what's passed to Layer's init.
        base_block_primary_params = {'embed_dim', 'num_heads', 'ff_dim', 'rate', 'seed'}
        layer_kwargs = {k: v for k, v in kwargs.items() if k not in base_block_primary_params}

        tf.keras.layers.Layer.__init__(self, name=name, **layer_kwargs)

        # Now call BaseBlock's init. It will use self.name (set by Layer's init)
        # and store its specific params in self.block_specific_config.
        BaseBlock.__init__(self, name=self.name, embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim, rate=rate, seed=seed, **kwargs)

        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.ff_dim = int(ff_dim)
        self.rate = float(rate)
        self.seed = seed # Can be None, int

        if self.num_heads <= 0:
            raise ValueError(f"num_heads ({self.num_heads}) must be positive.")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        self.head_dim = self.embed_dim // self.num_heads

        # Seed management for sub-layers
        self._current_seed_val = self.seed

        def get_next_seed():
            if self._current_seed_val is None:
                return None
            s = self._current_seed_val
            self._current_seed_val += 1
            return s

        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.head_dim,
            dropout=self.rate, # Dropout is applied to attention scores within MHA
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=get_next_seed())
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(self.ff_dim, activation="relu", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=get_next_seed())),
            tf.keras.layers.Dense(self.embed_dim, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=get_next_seed()))
        ], name=f"{self.name}_ffn")

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Dropout layers are applied *after* the MHA and FFN blocks respectively
        self.dropout_after_att = tf.keras.layers.Dropout(self.rate, seed=get_next_seed())
        self.dropout_after_ffn = tf.keras.layers.Dropout(self.rate, seed=get_next_seed())

    def call(self, inputs, training=None):
        """
        Forward pass for the TransformerBlock.

        Args:
            inputs: Input tensor. Shape `(batch_size, sequence_length, embed_dim)`.
            training (bool, optional): Indicates if the layer should behave in training mode (e.g., for dropout).

        Returns:
            Output tensor. Shape `(batch_size, sequence_length, embed_dim)`.
        """
        # Multi-Head Attention sub-layer
        # MHA expects query, value, key. For self-attention, these are all `inputs`.
        # The `training` argument is passed to MHA for its internal dropout.
        attn_output = self.att(query=inputs, value=inputs, key=inputs, training=training)
        # External dropout after MHA, as is common in some Transformer designs
        attn_output = self.dropout_after_att(attn_output, training=training)
        # Residual connection and Layer Normalization
        out1 = self.layernorm1(inputs + attn_output)

        # Feed-Forward Network sub-layer
        # The Sequential FFN also needs the `training` flag for its Dense layers if they had batchnorm or dropout (not in this case).
        ffn_output = self.ffn(out1, training=training)
        # External dropout after FFN
        ffn_output = self.dropout_after_ffn(ffn_output, training=training)
        # Residual connection and Layer Normalization
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

    def build(self, input_shape):
        """Standard Keras build method."""
        # Ensures sub-layers are built. MHA and Dense layers in FFN handle their own build.
        # LayerNorm also builds based on input shape.
        super(TransformerBlock, self).build(input_shape)

    def build_layer(self, input_shape=None):
        """
        Builds the layer structure (from BaseBlock interface).
        This method ensures the Keras layer is built, making it ready to be called.

        Args:
            input_shape (optional): The input shape for which to build the layer.

        Returns:
            self: The layer instance itself.
        """
        if input_shape and not self.built:
            self.build(input_shape)
        # If input_shape is None, Keras will build it on first call with actual data.
        return self

    def get_config(self):
        """Keras Layer's standard method for serialization."""
        # Start with Layer's config
        config = super(TransformerBlock, self).get_config()

        # Update with parameters specific to this TransformerBlock
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate,
            'seed': self.seed
        })

        # Add any other relevant kwargs that were passed in via **kwargs
        # and stored by BaseBlock in self.block_specific_config,
        # excluding those already handled or internal to Layer.
        if hasattr(self, 'block_specific_config'):
            for k, v in self.block_specific_config.items():
                # Avoid overwriting primary params or standard Layer config keys already set
                if k not in config and k not in ['name', 'trainable', 'dtype']:
                    config[k] = v
        return config
