import tensorflow as tf
from alpha_architecture.blocks.base_block import BaseBlock

class ReshapeBlock(BaseBlock, tf.keras.layers.Layer):
    """
    A block that reshapes the input to a target shape. Wraps tf.keras.layers.Reshape.
    Inherits from BaseBlock and tf.keras.layers.Layer.
    """
    def __init__(self, target_shape: tuple, name: str = 'ReshapeBlock', **kwargs):
        """
        Initializes the ReshapeBlock.

        Args:
            target_shape (tuple): The target shape for reshaping (excluding batch size).
            name (str, optional): Name of the block/layer. Defaults to 'ReshapeBlock'.
            **kwargs: Additional keyword arguments.
        """
        # Layer init: filter out 'target_shape' as it's not a standard Layer arg for __init__
        layer_kwargs = {k: v for k, v in kwargs.items() if k not in ['name', 'target_shape']}
        tf.keras.layers.Layer.__init__(self, name=name, **layer_kwargs)

        # BaseBlock init: pass 'target_shape' so it's in block_specific_config
        BaseBlock.__init__(self, name=self.name, target_shape=target_shape, **kwargs)

        self.target_shape = target_shape
        self.reshape_layer = tf.keras.layers.Reshape(self.target_shape)

    def call(self, inputs, training=None): # training arg for API consistency
        """
        Forward pass for the ReshapeBlock.

        Args:
            inputs: Input tensor.
            training (bool, optional): Unused by Reshape, but included for API consistency.

        Returns:
            Output tensor, reshaped.
        """
        return self.reshape_layer(inputs)

    def build(self, input_shape):
        """Standard Keras build method."""
        super(ReshapeBlock, self).build(input_shape)

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
        config = super(ReshapeBlock, self).get_config()
        config.update({'target_shape': self.target_shape})

        if hasattr(self, 'block_specific_config'):
            for k, v in self.block_specific_config.items():
                # Avoid overwriting standard/primary config keys
                if k not in config and k not in ['name', 'trainable', 'dtype', 'target_shape']:
                    config[k] = v
        return config
