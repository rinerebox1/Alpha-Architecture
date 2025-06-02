import tensorflow as tf
from alpha_architecture.blocks.base_block import BaseBlock

class Conv1DBlock(BaseBlock):
    """
    A block representing a Keras Conv1D layer.
    """
    def __init__(self, filters: int, kernel_size: int, activation: str = 'relu', name: str = 'Conv1D', **kwargs):
        """
        Initializes the Conv1DBlock.

        Args:
            filters (int): Number of output filters in the convolution.
            kernel_size (int): Length of the 1D convolution window.
            activation (str, optional): Activation function to use. Defaults to 'relu'.
            name (str, optional): Name of the block. Defaults to 'Conv1D'.
            **kwargs: Additional keyword arguments passed to the base class.
        """
        super().__init__(name=name, filters=filters, kernel_size=kernel_size, activation=activation, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation

    def build_layer(self, input_shape) -> tf.keras.layers.Layer:
        """
        Builds and returns a Keras Conv1D layer.

        Args:
            input_shape: The shape of the input tensor. Conv1D layers typically require a 3D input.

        Returns:
            A tf.keras.layers.Conv1D instance.
        """
        # The agent generating the architecture needs to ensure input_shape is compatible (e.g., 3D).
        return tf.keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation=self.activation, padding='same', name=self.name)
