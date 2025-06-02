import tensorflow as tf
from alpha_architecture.blocks.base_block import BaseBlock

class GlobalAveragePooling1DBlock(BaseBlock, tf.keras.layers.Layer):
    """
    A block that applies Global Average Pooling 1D. Wraps tf.keras.layers.GlobalAveragePooling1D.
    Inherits from BaseBlock and tf.keras.layers.Layer.
    """
    def __init__(self, name: str = 'GlobalAveragePooling1DBlock', **kwargs):
        """
        Initializes the GlobalAveragePooling1DBlock.

        Args:
            name (str, optional): Name of the block/layer. Defaults to 'GlobalAveragePooling1DBlock'.
            **kwargs: Additional keyword arguments.
        """
        # Layer init
        layer_kwargs = {k: v for k, v in kwargs.items() if k != 'name'}
        tf.keras.layers.Layer.__init__(self, name=name, **layer_kwargs)

        # BaseBlock init
        BaseBlock.__init__(self, name=self.name, **kwargs)

        self.pooling_layer = tf.keras.layers.GlobalAveragePooling1D()

    def call(self, inputs, training=None): # training arg for API consistency
        """
        Forward pass for the GlobalAveragePooling1DBlock.

        Args:
            inputs: Input tensor.
            training (bool, optional): Unused, but included for API consistency.

        Returns:
            Output tensor after global average pooling.
        """
        return self.pooling_layer(inputs)

    def build(self, input_shape):
        """Standard Keras build method."""
        super(GlobalAveragePooling1DBlock, self).build(input_shape)

    def build_layer(self, input_shape=None):
        """
        Builds the layer structure (from BaseBlock interface).

        Args:
            input_shape (optional): The input shape for which to build the layer.

        Returns:
            self: The layer instance itself.
        """
        if input_shape and not self.built:
            self.build(input_shape)
        return self

    def get_config(self):
        """Keras Layer's standard method for serialization."""
        config = super(GlobalAveragePooling1DBlock, self).get_config()

        if hasattr(self, 'block_specific_config'):
            for k, v in self.block_specific_config.items():
                if k not in config and k not in ['name', 'trainable', 'dtype']:
                    config[k] = v
        return config
