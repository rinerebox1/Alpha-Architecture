import tensorflow as tf
from alpha_architecture.blocks.base_block import BaseBlock

class FlattenBlock(BaseBlock, tf.keras.layers.Layer):
    """
    A block that flattens the input. Wraps tf.keras.layers.Flatten.
    Inherits from BaseBlock for architectural generation and tf.keras.layers.Layer for Keras integration.
    """
    def __init__(self, name: str = 'FlattenBlock', **kwargs):
        """
        Initializes the FlattenBlock.

        Args:
            name (str, optional): Name of the block/layer. Defaults to 'FlattenBlock'.
            **kwargs: Additional keyword arguments.
        """
        # Call Layer's init first to set self.name, etc.
        # Filter out BaseBlock specific kwargs if any were accidentally passed that Layer wouldn't expect.
        # For Flatten, there are no unique params beyond 'name'.
        layer_kwargs = {k: v for k, v in kwargs.items() if k != 'name'} # Example, though Flatten has few specific args
        tf.keras.layers.Layer.__init__(self, name=name, **layer_kwargs)

        # Call BaseBlock's init using self.name set by Layer's init.
        # Pass all original kwargs (including name and any specific params for BaseBlock)
        BaseBlock.__init__(self, name=self.name, **kwargs)

        self.flatten_layer = tf.keras.layers.Flatten()

    def call(self, inputs, training=None): # training arg for API consistency
        """
        Forward pass for the FlattenBlock.

        Args:
            inputs: Input tensor.
            training (bool, optional): Unused by Flatten, but included for API consistency.

        Returns:
            Output tensor, flattened.
        """
        return self.flatten_layer(inputs)

    def build(self, input_shape):
        """Standard Keras build method."""
        super(FlattenBlock, self).build(input_shape)
        # self.flatten_layer.build(input_shape) # Not strictly necessary as it's built in its __init__ or first call

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
        config = super(FlattenBlock, self).get_config()
        # Add any specific params from BaseBlock's kwargs if they were intended for config
        if hasattr(self, 'block_specific_config'):
            for k, v in self.block_specific_config.items():
                # Avoid overwriting standard Layer config keys or already handled params
                if k not in config and k not in ['name', 'trainable', 'dtype']:
                    config[k] = v
        return config
