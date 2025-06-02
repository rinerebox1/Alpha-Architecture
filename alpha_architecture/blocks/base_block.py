from abc import ABC, abstractmethod
import tensorflow as tf

class BaseBlock(ABC):
    """
    Abstract base class for all neural network blocks.
    Each block represents a configurable part of a neural network architecture.
    """
    def __init__(self, name: str, **kwargs):
        """
        Initializes the BaseBlock.

        Args:
            name (str): A human-readable name for the block type (e.g., "Dense", "LSTM").
            **kwargs: Block-specific configurations.
        """
        self.name = name
        self.block_specific_config = kwargs

    @abstractmethod
    def build_layer(self, input_shape) -> tf.keras.layers.Layer:
        """
        Builds and returns a Keras layer compatible with the given input_shape.
        This method must be implemented by subclasses.

        Args:
            input_shape: The shape of the input tensor that this layer will receive.

        Returns:
            A tf.keras.layers.Layer instance.
        """
        pass

    @property
    def config(self) -> dict:
        """
        Returns the configuration of the block.

        Returns:
            A dictionary containing the block's name and other specific configurations.
        """
        return {
            'name': self.name,
            **self.block_specific_config
        }
