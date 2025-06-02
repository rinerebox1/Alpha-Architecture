import random
import tensorflow as tf
from alpha_architecture.blocks.base_block import BaseBlock
from alpha_architecture.blocks.dense_block import DenseBlock
from alpha_architecture.blocks.lstm_block import LSTMBlock
from alpha_architecture.blocks.conv1d_block import Conv1DBlock
from alpha_architecture.blocks.transformer_block import TransformerBlock
from alpha_architecture.blocks.flatten_block import FlattenBlock
from alpha_architecture.blocks.global_average_pooling1d_block import GlobalAveragePooling1DBlock

class Agent:
    """
    The Agent class is responsible for generating neural network architectures
    using a predefined set of available building blocks.
    """
    def __init__(self, available_block_prototypes: list, input_shape: tuple, seed: int = None):
        """
        Initializes the Agent.

        Args:
            available_block_prototypes (list): A list of block class constructors
                                               (e.g., [DenseBlock, LSTMBlock]).
            input_shape (tuple): The input shape for the models to be generated
                                 (e.g., (timesteps, features)).
            seed (int, optional): A seed for the agent's random number generators
                                  to ensure reproducibility. Defaults to None.
        """
        if not available_block_prototypes:
            raise ValueError("available_block_prototypes cannot be empty.")
        self.available_block_prototypes = available_block_prototypes
        self.input_shape = input_shape

        self.agent_seed = seed
        self._internal_block_seed_counter = 0
        if self.agent_seed is not None:
            random.seed(self.agent_seed) # Seed Python's random for choices, etc.
            # Note: tf.random.set_seed is global; set it in the script if needed for TF ops outside agent.
            # Keras initializers within blocks can take specific seeds.

    def _get_next_block_seed(self) -> int | None:
        """Generates a new seed for a block, derived from the agent's main seed."""
        if self.agent_seed is None:
            return None
        # Create a sequence of seeds derived from the agent's main seed
        block_seed = self.agent_seed + self._internal_block_seed_counter
        self._internal_block_seed_counter += 10 # Increment by a larger step to ensure seeds are well apart
        return block_seed

    def _create_block_instance(self, block_prototype, unique_name: str, current_output_shape=None) -> BaseBlock | None:
        """
        Creates an instance of a block with randomly chosen or context-aware hyperparameters.

        Args:
            block_prototype: The class constructor of the block to instantiate.
            unique_name (str): A unique name for this block instance.
            current_output_shape (tuple, optional): The output shape of the previous layer.
                                                   Needed for context-aware blocks like Transformer.

        Returns:
            An instance of a BaseBlock subclass, or None if the block is not suitable.
        """
        block_seed = self._get_next_block_seed()

        if block_prototype == DenseBlock:
            # Dense can handle various input shapes.
            # Seed could be passed if DenseBlock's Keras layers were set up to use it.
            return DenseBlock(
                units=random.choice([32, 64, 128]),
                activation=random.choice(['relu', 'tanh', 'sigmoid']),
                name=unique_name
            )
        elif block_prototype == LSTMBlock:
            # LSTM expects 3D input typically. Agent should ensure this or add Reshape.
            # For now, we assume input is appropriate or it's handled by model construction logic.
            return LSTMBlock(
                units=random.choice([32, 64, 128]),
                return_sequences=True, # Keep True for stacking by default
                name=unique_name
            )
        elif block_prototype == Conv1DBlock:
            # Conv1D also expects 3D input.
            return Conv1DBlock(
                filters=random.choice([32, 64, 128]),
                kernel_size=random.choice([3, 5, 7]),
                activation=random.choice(['relu', 'tanh']),
                name=unique_name
            )
        elif block_prototype == TransformerBlock:
            if current_output_shape is None or len(current_output_shape) != 3:
                # print(f"TransformerBlock {unique_name} requires 3D input, got {current_output_shape}. Skipping.")
                return None

            embed_dim = current_output_shape[-1]
            # Define allowed embed_dim values for Transformer; must match features of input.
            # These are examples; in a real scenario, might need more flexible handling or Reshape.
            valid_transformer_embed_dims = [16, 32, 48, 64, 96, 128]
            if embed_dim not in valid_transformer_embed_dims:
                # print(f"TransformerBlock {unique_name} embed_dim {embed_dim} not in valid list. Skipping.")
                return None

            possible_num_heads = [h for h in [2, 4, 8] if embed_dim % h == 0 and h > 0]
            if not possible_num_heads:
                # print(f"TransformerBlock {unique_name} no valid num_heads for embed_dim {embed_dim}. Skipping.")
                return None
            num_heads = random.choice(possible_num_heads)

            # ff_dim typically larger than embed_dim
            ff_dim = random.choice([embed_dim * 1, embed_dim * 2, embed_dim * 4])
            rate = random.choice([0.0, 0.1, 0.2]) # Add some randomness to dropout

            return TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                rate=rate,
                name=unique_name,
                seed=block_seed
            )
        elif block_prototype == FlattenBlock:
            # Flatten is generally applicable if input is not already 1D (excluding batch).
            if current_output_shape and len(current_output_shape) <= 2: # (batch, features) or (batch, 1)
                return None # Already flat or effectively flat
            return FlattenBlock(name=unique_name)
        elif block_prototype == GlobalAveragePooling1DBlock:
            if current_output_shape is None or len(current_output_shape) != 3:
                # print(f"GlobalAveragePooling1DBlock {unique_name} requires 3D input. Skipping.")
                return None
            return GlobalAveragePooling1DBlock(name=unique_name)
        # ReshapeBlock would need target_shape as a parameter, making it harder for random generation
        # without more sophisticated meta-parameter generation. Skipping for now in _create_block_instance.
        else:
            # Fallback for unknown block types, though ideally the list is controlled
            raise ValueError(f"Unsupported block prototype: {block_prototype}")

    def generate_random_architecture(self, max_depth: int, output_units: int = 1, output_activation: str = 'linear') -> tf.keras.Sequential | None:
        if max_depth <= 0:
            raise ValueError("max_depth must be positive.")

        model_name = f"random_arch_{random.randint(1000, 9999)}"
        model = tf.keras.Sequential(name=model_name)
        model.add(tf.keras.Input(shape=self.input_shape, name="input_layer"))

        num_layers = random.randint(1, max_depth)
        actual_layers_added = 0

        for i in range(num_layers):
            block_prototype = random.choice(self.available_block_prototypes)

            # Ensure block name uniqueness within the model
            # Use class name (lowercase, no 'block') and loop index
            block_class_name = block_prototype.__name__.replace('Block', '').lower()
            layer_name = f"{block_class_name}_{i}"

            current_model_output_shape = model.output_shape

            block_instance = self._create_block_instance(
                block_prototype,
                unique_name=layer_name,
                current_output_shape=current_model_output_shape
            )

            if block_instance is None:
                # print(f"Could not create instance for {layer_name} with input shape {current_model_output_shape}, trying next.")
                continue # Try adding a different block

            # build_layer for Keras Layers (like TransformerBlock, FlattenBlock) ensures they are built.
            # For non-Keras Layer blocks, it returns the Keras layer.
            keras_layer_to_add = block_instance.build_layer(None) # input_shape not needed after Input layer for Keras layers

            model.add(keras_layer_to_add)
            actual_layers_added +=1
            # print(f"Added {layer_name} (type: {block_prototype.__name__}). New model output shape: {model.output_shape}")


        if actual_layers_added == 0 and num_layers > 0 :
            # This might happen if no blocks were suitable for the input shape or subsequent shapes.
            # print(f"Failed to add any hidden layers to model {model_name}.")
            # Return None or a very simple model (e.g. just input -> output dense)
            # For now, let it proceed, it might just have an Input and Output dense layer.
            pass

        # Output Layer Handling
        if len(model.output_shape) > 2:
            # If last layer was sequence returning (e.g. LSTM, Conv1D, Transformer)
            # and Flatten wasn't the last chosen block.
            # Check if last layer is already Flatten or GlobalAveragePooling1D
            last_layer_is_pooling_or_flatten = False
            if model.layers:
                last_keras_layer = model.layers[-1]
                if isinstance(last_keras_layer, (tf.keras.layers.Flatten, tf.keras.layers.GlobalAveragePooling1D)):
                    last_layer_is_pooling_or_flatten = True

            if not last_layer_is_pooling_or_flatten:
                 model.add(tf.keras.layers.Flatten(name="final_flatten"))

        model.add(tf.keras.layers.Dense(units=output_units, activation=output_activation, name='output_layer'))

        return model
