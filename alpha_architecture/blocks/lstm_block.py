import tensorflow as tf
from alpha_architecture.blocks.base_block import BaseBlock

class LSTMBlock(BaseBlock):
    """
    A block representing a Keras LSTM layer.
    """
    def __init__(self, units: int, return_sequences: bool = False, name: str = 'LSTM', **kwargs):
        """
        Initializes the LSTMBlock.

        Args:
            units (int): Number of units in the LSTM layer.
            return_sequences (bool, optional): Whether to return the last output in the output sequence,
                                               or the full sequence. Defaults to False.
            name (str, optional): Name of the block. Defaults to 'LSTM'.
            **kwargs: Additional keyword arguments passed to the base class.
        """
        super().__init__(name=name, units=units, return_sequences=return_sequences, **kwargs)
        self.units = units
        self.return_sequences = return_sequences

    def build_layer(self, input_shape) -> tf.keras.layers.Layer:
        """
        Builds and returns a Keras LSTM layer.

        Args:
            input_shape: The shape of the input tensor. LSTM layers typically require a 3D input.

        Returns:
            A tf.keras.layers.LSTM instance.
        """
        # The agent generating the architecture needs to ensure input_shape is compatible (e.g., 3D).
        return tf.keras.layers.LSTM(units=self.units, return_sequences=self.return_sequences, name=self.name)
