# scripts/run_experiment.py
import tensorflow as tf
import numpy as np
import random
import sys
import os

# Add the project root to the Python path to allow imports from alpha_architecture
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from alpha_architecture.agent import Agent
from alpha_architecture.blocks.dense_block import DenseBlock
from alpha_architecture.blocks.lstm_block import LSTMBlock
from alpha_architecture.blocks.conv1d_block import Conv1DBlock
from alpha_architecture.blocks.transformer_block import TransformerBlock
from alpha_architecture.blocks.flatten_block import FlattenBlock
from alpha_architecture.blocks.global_average_pooling1d_block import GlobalAveragePooling1DBlock
# ReshapeBlock is not currently added to AVAILABLE_BLOCK_PROTOTYPES as its target_shape needs dynamic configuration.

# Set global seeds for reproducibility for operations outside the Agent's direct control (e.g., data generation, TF ops)
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
tf.random.set_seed(GLOBAL_SEED) # Sets TensorFlow's global random seed
random.seed(GLOBAL_SEED) # Python's random, for any top-level script choices

def generate_synthetic_data(n_samples=1000, timesteps=10, n_features=1):
    print(f"Generating synthetic data: {n_samples} samples, {timesteps} timesteps, {n_features} features.")
    X = np.zeros((n_samples, timesteps, n_features), dtype=np.float32)
    for i in range(n_samples):
        start = np.random.rand() * 10
        X[i, :, 0] = np.sin(np.linspace(start, start + np.pi * 2 * (timesteps / 10.0), timesteps)) + \
                       np.random.normal(0, 0.1, timesteps)

    y = np.mean(X[:, -3:, :], axis=1, keepdims=True).astype(np.float32)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    return X, y

def main():
    # Experiment Configuration
    TIMESTEPS = 20
    # For TransformerBlock, N_FEATURES should be one of [16, 32, 48, 64, 96, 128] if it's the first layer.
    # Or, a Dense/Conv layer must precede Transformer to adjust features to a valid embed_dim.
    # Let's use N_FEATURES = 16 to allow Transformer as a potential first layer.
    N_FEATURES = 16
    INPUT_SHAPE = (TIMESTEPS, N_FEATURES)

    AVAILABLE_BLOCK_PROTOTYPES = [
        DenseBlock,
        LSTMBlock,
        Conv1DBlock,
        TransformerBlock,
        FlattenBlock, # Usually added by agent logic if needed, but can be chosen.
        GlobalAveragePooling1DBlock
    ]

    NUM_ARCHITECTURES_TO_TEST = 5 # Increased slightly to see more variety
    MAX_MODEL_DEPTH = 4
    EPOCHS = 5
    BATCH_SIZE = 32

    print("Initializing Alpha-Architecture Agent...")
    agent = Agent(
        available_block_prototypes=AVAILABLE_BLOCK_PROTOTYPES,
        input_shape=INPUT_SHAPE,
        seed=GLOBAL_SEED # Pass the global seed to the agent for its internal RNGs
    )

    print("Generating and splitting synthetic data...")
    # Adjusted n_samples for quicker run with increased N_FEATURES
    X, y = generate_synthetic_data(n_samples=500, timesteps=TIMESTEPS, n_features=N_FEATURES)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"Training data: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
    print(f"Test data: X_test shape {X_test.shape}, y_test shape {y_test.shape}")

    results = []

    for i in range(NUM_ARCHITECTURES_TO_TEST):
        print(f"\n--- Testing Architecture {i+1}/{NUM_ARCHITECTURES_TO_TEST} ---")
        try:
            # Output units for y: y has N_FEATURES if it's multi-output, or 1 if single value.
            # Our y is (num_samples, 1) due to np.mean(..., keepdims=True) and original n_features=1 for y target.
            # However, X now has N_FEATURES. If y is still mean of X, y will be (num_samples, N_FEATURES).
            # Correcting y generation for N_FEATURES > 1:
            # y = np.mean(X, axis=(1,2), keepdims=True).astype(np.float32) # Example: mean of all elements
            # For this example, let's make y related to one of the features if N_FEATURES > 1
            # The current y generation: np.mean(X[:, -3:, :], axis=1, keepdims=True) will produce y of shape (n_samples, N_FEATURES)
            # So, output_units should be N_FEATURES.

            model = agent.generate_random_architecture(
                max_depth=MAX_MODEL_DEPTH,
                output_units=N_FEATURES, # y_train/y_test are of shape (None, N_FEATURES)
                output_activation='linear'
            )

            if model is None or not model.layers: # Check if model generation failed
                print(f"Architecture {i+1} generation failed. Skipping.")
                results.append({'architecture_name': f"Failed_Arch_{i+1}", 'loss': float('inf'), 'mae': float('inf'), 'model_summary': None})
                continue

            print("Generated Model Summary:")
            model.summary(line_length=100) # Adjusted line_length for wider summaries

            print("Compiling model...")
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss='mse', metrics=['mae']) # Reduced LR

            print("Training model...")
            history = model.fit(
                X_train, y_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_data=(X_test, y_test),
                verbose=1
            )

            print("Evaluating model...")
            loss, mae = model.evaluate(X_test, y_test, verbose=0)
            print(f"Architecture {i+1} ('{model.name}') - Test Loss (MSE): {loss:.4f}, Test MAE: {mae:.4f}")
            results.append({'architecture_name': model.name, 'loss': loss, 'mae': mae, 'model_summary': model.to_json()})
        except Exception as e:
            print(f"Error generating/training architecture {i+1}: {e}")
            import traceback
            traceback.print_exc()
            results.append({'architecture_name': f"Error_Arch_{i+1}_{type(e).__name__}", 'loss': float('inf'), 'mae': float('inf'), 'model_summary': None})

    print("\n--- Experiment Summary ---")
    if results:
        for res in sorted(results, key=lambda x: x['loss']):
            print(f"Architecture: {res['architecture_name']}, Test MSE: {res['loss']:.4f}, Test MAE: {res['mae']:.4f}")
    else:
        print("No results to display.")

if __name__ == '__main__':
    main()
