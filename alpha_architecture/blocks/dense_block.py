import tensorflow as tf
from alpha_architecture.blocks.base_block import BaseBlock

class DenseBlock(BaseBlock):
    """
    A block representing a Keras Dense layer.
    """
    def __init__(self, units: int, activation: str = 'relu', name: str = 'Dense', **kwargs):
        """
        Initializes the DenseBlock.

        Args:
            units (int): Number of units in the dense layer.
            activation (str, optional): Activation function to use. Defaults to 'relu'.
            name (str, optional): Name of the block. Defaults to 'Dense'.
            **kwargs: Additional keyword arguments passed to the base class.
        """
        super().__init__(name=name, units=units, activation=activation, **kwargs)
        self.units = units
        self.activation = activation

    def build_layer(self, input_shape=None) -> tf.keras.layers.Layer:
        """
        Builds and returns a Keras Dense layer.

        Args:
            input_shape (optional): The shape of the input tensor.
                                   Not strictly required for Dense if not the first layer.

        Returns:
            A tf.keras.layers.Dense instance.
        """
        return tf.keras.layers.Dense(units=self.units, activation=self.activation, name=self.name)
