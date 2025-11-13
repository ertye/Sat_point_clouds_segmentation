import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    Activation,
    SpatialDropout1D,
    Dense,
    Add,
)
from tensorflow.keras.models import Model
import numpy as np


def check_for_gpu():
    """
    Checks if TensorFlow can detect a GPU (CUDA device).
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Restrict TensorFlow to only use the first GPU
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            # Enable memory growth to avoid allocating all GPU memory at once
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print(f"Success! TensorFlow is using GPU: {gpus[0].name}")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    else:
        print("Warning: TensorFlow did not find a GPU. Running on CPU.")

# --- 1. Define the core TCN Residual Block ---
# This function is the Keras equivalent of the PyTorch 'TemporalBlock'
# It uses the parameters from the paper (BN, SpatialDropout)
def temporal_block(input_tensor, n_outputs, kernel_size, dilation, dropout_rate):
    """
    Creates a TCN residual block as described in the paper.
    
    Args:
        input_tensor: The input tensor from the previous layer.
        n_outputs (int): The number of filters (paper's "hidden size of 64").
        kernel_size (int): The size of the convolution kernel.
        dilation (int): The dilation rate.
        dropout_rate (float): The dropout rate (paper's "0.3").
        
    Returns:
        A Keras tensor representing the output of the block.
    """
    
    # Get the number of input channels for the residual connection
    input_channels = input_tensor.shape[-1]
    
    # --- First Convolutional Layer in the Block ---
    x = Conv1D(
        filters=n_outputs,
        kernel_size=kernel_size,
        dilation_rate=dilation,
        padding='causal',  # 'causal' handles the padding just like Chomp1d
    )(input_tensor)
    
    # Paper specifies BatchNormalization
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Paper specifies "spatial dropout layer with a dropout rate of 0.3"
    x = SpatialDropout1D(dropout_rate)(x)

    # --- Second Convolutional Layer in the Block ---
    x = Conv1D(
        filters=n_outputs,
        kernel_size=kernel_size,
        dilation_rate=dilation,
        padding='causal',
    )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SpatialDropout1D(dropout_rate)(x)

    # --- Residual Connection ---
    # The PyTorch code has a 'downsample' layer if n_inputs != n_outputs
    # We do the same here with a 1x1 convolution
    if input_channels != n_outputs:
        # Projection shortcut (1x1 conv) to match dimensions
        residual = Conv1D(n_outputs, kernel_size=1)(input_tensor)
    else:
        # Identity shortcut
        residual = input_tensor

    # Add the residual connection to the block's output
    x = Add()([x, residual])
    return Activation('relu')(x)


# --- 2. Define the Full TCN Model (ExpressEar) ---
def create_express_ear_tcn(
    seq_length,
    num_features=12,
    num_au=5,
    hidden_units=64,
    kernel_size=2,
    dropout=0.3
):
    """
    Builds the complete ExpressEar TCN model based on the paper.

    Args:
        seq_length (int): The length of the input IMU sequences.
        num_features (int): Number of input features (12-axis L+R).
        num_au (int): Number of output Action Units (classes).
        hidden_units (int): Hidden size for each TCN layer (64).
        kernel_size (int): Kernel size (2, from PyTorch code).
        dropout (float): Dropout rate (0.3).
    
    Returns:
        A compiled Keras Model.
    """
    
    # Dilations list from the paper's diagram
    dilations = [1, 2, 4, 8, 16, 32]
    
    # --- Input Layer ---
    input_layer = Input(shape=(seq_length, num_features))
    
    x = input_layer
    
    # --- Stack Temporal Blocks ---
    # This creates the "12-layer" TCN (6 blocks * 2 conv layers/block)
    for dilation_rate in dilations:
        x = temporal_block(
            x,
            n_outputs=hidden_units,
            kernel_size=kernel_size,
            dilation=dilation_rate,
            dropout_rate=dropout
        )

    # --- Classification Head ---
    # The output 'x' is still a sequence (batch, seq_length, 64).
    # For classification, we typically take the *last* time step's output.
    x = x[:, -1, :]  # Shape is now (batch, 64)

    # --- Output Layer ---
    # Paper: "apply the sigmoid activation" for "multi-label classification"
    output_layer = Dense(num_au, activation='sigmoid')(x)

    # --- Build and Compile Model ---
    model = Model(inputs=input_layer, outputs=output_layer)

    # Paper: "Adam optimizer (learning rate: 0.01) and binary cross-entropy"
    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.01, weight_decay=1e-5)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy', # For multi-label classification
        metrics=['accuracy'] # 'accuracy' is good, 'AUC' is also common
    )
    
    return model


# --- Custom Callback combining complex patience and saving logic ---
class CustomEarlyStopping(tf.keras.callbacks.Callback):
    """
    Custom early stopping callback that replicates the complex PyTorch logic
    Combines:
    1. Validation loss complex patience
    2. High accuracy stopping
    3. Saving on every epoch
    4. Saving on stop
    """
    def __init__(self, patience=5, accuracy_threshold=0.99990, model_save_prefix=""):
        super(CustomEarlyStopping, self).__init__()
        
        # Parameters from __init__
        self.patience_max = patience
        self.accuracy_threshold = accuracy_threshold
        self.model_save_prefix = model_save_prefix
        
        # Internal state variables from the 'for' loop logic
        self.patience_i = 0
        self.patience_diff = 0.0
        self.patience_diff_1 = 0.0
        self.valid_loss_old = 99999.0  # Initialize with a large number
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get('val_loss')
        current_val_acc = logs.get('val_accuracy')

        # --- 1. Complex Patience Logic ---
        if self.valid_loss_old < current_val_loss:
            # Loss got worse
            self.patience_i += 1
            self.patience_diff += current_val_loss - self.valid_loss_old
            # Safely calculate the average deterioration
            if self.patience_i > 0:
                self.patience_diff_1 = self.patience_diff / self.patience_i
            else:
                print('Error happened in CustomEarlyStopping on_epoch_end() p1!\n')
                self.patience_diff_1 = 0.001
                self.patience_i=999
        else:
            # Loss improved or stayed same
            self.patience_diff += current_val_loss - self.valid_loss_old
            if self.patience_diff < 0:
                # Accumulated deterioration has been completely offset, reset all counters
                self.patience_diff = 0
                self.patience_diff_1 = 0
                self.patience_i = 0
            else:
                # Loss improved, but did not completely offset accumulated deterioration
                # (Fix for potential NaN/ZeroDivisionError)
                if self.patience_diff_1 > 0:
                    # Recalculate how much patience is still needed
                    self.patience_i = int(np.ceil(self.patience_diff / self.patience_diff_1))
                    if self.patience_i > 0:
                        self.patience_diff_1 = self.patience_diff / self.patience_i
                    elif self.patience_i == 0:
                        # If self.patience_i is 0, reset
                        self.patience_diff = 0
                        self.patience_diff_1 = 0
                    else:
                        # If self.patience_i < 0
                        print('Error happened in CustomEarlyStopping on_epoch_end() p3!\n')
                        self.patience_diff_1 = 0.001
                        self.patience_i=999
                else:
                    print('Error happened in CustomEarlyStopping on_epoch_end() p2!\n')
                    self.patience_diff_1 = 0.001
                    self.patience_i=999

        # --- 2. Check Stopping Conditions ---
        
        # A. High Accuracy Stop
        if current_val_acc is not None and current_val_acc > self.accuracy_threshold:
            print(f'\nEpoch {epoch+1}: 0.99990 accuracy reached - stopping training ###### ')
            self.model.stop_training = True
            self.stopped_epoch = epoch
            save_path = f'{self.model_save_prefix}_epoch{epoch+1}.h5'
            print(f'Saving final model to {save_path}')
            self.model.save(save_path)
            return

        # B. Complex Patience Stop
        if self.patience_i >= self.patience_max:
            print(f'\nEpoch {epoch+1}: Early stopping triggered (patience_max) after {self.patience_i} epochs without net improvement ###### ')
            self.model.stop_training = True
            self.stopped_epoch = epoch
            save_path = f'{self.model_save_prefix}_epoch{epoch+1}.h5'
            print(f'Saving final model to {save_path}')
            self.model.save(save_path)
            return
            
        # --- 3. Save model after every epoch (from original logic) ---
        if epoch >= 1:
            save_path = f'{self.model_save_prefix}_epoch{epoch+1}.h5'
            # print(f'\nEpoch {epoch+1}: Saving model to {save_path}') # Optional: uncomment for verbose saving
            self.model.save(save_path)
        
        # --- 4. Update the old loss value ---
        self.valid_loss_old = current_val_loss

class LossHistory(tf.keras.callbacks.Callback):
    """Callback to log training history to file in the original format"""
    def __init__(self, file_path):
        super(LossHistory, self).__init__()
        self.file_path = file_path
        with open(self.file_path, 'w') as f:
            f.write('Epoch, Train Loss, Valid Loss, Valid Acc\n')
    
    def on_epoch_end(self, epoch, logs=None):
        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        val_acc = logs.get('val_accuracy')
        
        with open(self.file_path, 'a') as f:
            f.write(f' {epoch + 1}, {train_loss:.6f}, {val_loss:.6f}, {val_acc:.6f}\n')
# --- 3. Main execution: Training and Testing ---
if __name__ == "__main__":
    
    # --- Step 1: Verify GPU (CUDA) is active ---
    check_for_gpu()
    
    # --- Step 2: Define Model & Data Parameters ---
    # (Using dummy values since we don't have the real .mat files)
    
    # Input data shape
    NUM_SAMPLES_TRAIN = 1000
    NUM_SAMPLES_TEST = 200
    SEQ_LENGTH = 100      # Receptive field is 64, so length must be >= 64
    NUM_FEATURES = 12     # From paper: "12 axis (L+R)"
    
    # Output data shape
    NUM_CLASSES = 5       # Example: 5 different Action Units (AUs)

    # --- Step 3: Generate Dummy Data ---
    print("\nGenerating dummy data...")
    # X_train shape: (1000, 100, 12)
    X_train = np.random.rand(NUM_SAMPLES_TRAIN, SEQ_LENGTH, NUM_FEATURES).astype(np.float32)
    # y_train shape: (1000, 5) - Multi-label, so labels are 0 or 1
    y_train = np.random.randint(0, 2, size=(NUM_SAMPLES_TRAIN, NUM_CLASSES)).astype(np.float32)
    
    # X_test shape: (200, 100, 12)
    X_test = np.random.rand(NUM_SAMPLES_TEST, SEQ_LENGTH, NUM_FEATURES).astype(np.float32)
    # y_test shape: (200, 5)
    y_test = np.random.randint(0, 2, size=(NUM_SAMPLES_TEST, NUM_CLASSES)).astype(np.float32)
    
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    # --- Step 4: Create the Model ---
    print("\nCreating TCN model...")
    model = create_express_ear_tcn(
        seq_length=SEQ_LENGTH,
        num_features=NUM_FEATURES,
        num_au=NUM_CLASSES,
        hidden_units=64,     # From paper
        kernel_size=2,       # From TCN PyTorch code
        dropout=0.3          # From paper
    )
    
    model.summary()
    
    # --- Step 5: Setup Callbacks ---
    print("\nSetting up custom callbacks...")
    early_stopping = CustomEarlyStopping(
        patience=5,
        accuracy_threshold=0.99990,
        model_save_prefix='tcn_model' # Using a generic prefix
    )
    
    loss_history = LossHistory(file_path='Loss_History.txt')

    # --- Step 5: Train the Model ---
    # This .fit() command will AUTOMATICALLY use the GPU (CUDA)
    print("\n--- Starting Model Training (using GPU if available) ---")
    history = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=999,
        validation_split=0.1,  # Use 10% of training data for validation
        callbacks=[early_stopping, loss_history] # ADDED custom callbacks
    )
    print("--- Model Training Complete ---")

    # --- Step 6: Test (Evaluate) the Model ---
    print("\n--- Evaluating Model on Test Data ---")
    results = model.evaluate(X_test, y_test, batch_size=32)
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")

    # --- Step 7: Predict with the Model ---
    print("\n--- Making a Prediction ---")
    # Take one sample from the test set
    sample = X_test[0:1]  # Keep the batch dimension
    prediction = model.predict(sample)
    
    print(f"Input sample shape: {sample.shape}")
    print(f"Prediction (raw logits): {prediction}")
    print(f"Prediction (rounded): {(prediction > 0.5).astype(int)}")

    print(f"Actual Label: {y_test[0].astype(int)}")
