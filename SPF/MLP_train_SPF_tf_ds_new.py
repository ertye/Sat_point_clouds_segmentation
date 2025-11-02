import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses
import numpy as np
from pathlib import Path
import argparse

SPF = 10  # 3,4,8,10

def readConfigFiles():
    """Read configuration parameters from external files"""
    global SPF
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent

    # Read hidden size configuration
    file_path = parent_dir / 'hiddenSize.txt'
    with file_path.open('r', encoding='utf-8') as f:
        content = f.read()
    lines = content.split('\n')
    if SPF == 3:
        hidden_size = int(lines[1])
    elif SPF == 4:
        hidden_size = int(lines[3])
    elif SPF == 8:
        hidden_size = int(lines[5])
    elif SPF == 10:
        hidden_size = int(lines[7])

    # Read learning rate configuration
    file_path = parent_dir / 'learningRate.txt'
    with file_path.open('r', encoding='utf-8') as f:
        content = f.read()
    lines = content.split('\n')
    if SPF == 3:
        learningRate = float(lines[1])
    elif SPF == 4:
        learningRate = float(lines[3])
    elif SPF == 8:
        learningRate = float(lines[5])
    elif SPF == 10:
        learningRate = float(lines[7])        

    return hidden_size, learningRate

def read_points_from_txt(file_path):
    """Read point cloud data from txt file"""
    points = np.loadtxt(file_path, delimiter=',')
    return points

def create_mlp_model(input_size, hidden_size, output_size):
    """Create MLP model using Keras Sequential API"""
    model = keras.Sequential()
    
    # Input layer and hidden layers
    model.add(layers.Dense(int(hidden_size/2), input_shape=(input_size,)))
    model.add(layers.LeakyReLU(alpha=0.01))
    
    model.add(layers.Dense(hidden_size))
    model.add(layers.LeakyReLU(alpha=0.01))
    
    # Multiple hidden layers with same configuration
    for _ in range(6):  # fc3 to fc8 (6 layers)
        model.add(layers.Dense(hidden_size))
        model.add(layers.LeakyReLU(alpha=0.01))
    
    # Final layers
    model.add(layers.Dense(int(hidden_size/2)))
    model.add(layers.LeakyReLU(alpha=0.01))
    
    model.add(layers.Dense(6))
    model.add(layers.LeakyReLU(alpha=0.01))
    
    model.add(layers.Dense(output_size))
    model.add(layers.LeakyReLU(alpha=0.01))
    
    return model

def create_train_data():
    """Create training and validation datasets"""
    global SPF
    current_dir = Path(__file__).parent
    file_path = current_dir / 'SPF_Land_Water_Training.txt'    

    # Read point cloud data
    points = read_points_from_txt(file_path)
    if np.size(points, 1) == 15:
        SPF = SPF + 2
    print(f'SP{SPF}F: read_points_from_txt ###### ')

    # Select features based on SPF value and data dimensions
    if np.size(points, 1) == 15:
        if SPF == 5:
            input_data_2d = points[:, [4, 6, 8, 13, 14]]
        elif SPF == 6:
            input_data_2d = points[:, [3, 4, 6, 8, 13, 14]]
        elif SPF == 10:
            input_data_2d = points[:, [3, 4, 5, 6, 8, 9, 10, 11, 13, 14]]
        elif SPF == 12:
            input_data_2d = points[:, 3:]
    elif np.size(points, 1) == 13:
        if SPF == 3:
            input_data_2d = points[:, [4, 6, 8]]
        elif SPF == 4:
            input_data_2d = points[:, [3, 4, 6, 8]]
        elif SPF == 8:
            input_data_2d = points[:, [3, 4, 5, 6, 8, 9, 10, 11]]
        elif SPF == 10:
            input_data_2d = points[:, 3:]

    train_size = int(input_data_2d.shape[0] * 0.8)

    train_data = input_data_2d[1:train_size, :] 
    valid_data = input_data_2d[train_size:, :] 
    print(f'SP{SPF}F: input_data prepared ###### ')

    # Extract labels (first column)
    train_labels = points[1:train_size, 0]
    valid_labels = points[train_size:, 0]

    return train_data, train_labels, input_data_2d.shape[1], valid_data, valid_labels

class CustomEarlyStopping(keras.callbacks.Callback):
    """
    Custom early stopping callback that replicates the original PyTorch logic
    Combines validation loss patience and high accuracy stopping
    """
    def __init__(self, patience=5, accuracy_threshold=0.99990, model_save_prefix=""):
        super(CustomEarlyStopping, self).__init__()
        self.patience = patience
        self.accuracy_threshold = accuracy_threshold
        self.model_save_prefix = model_save_prefix
        self.best_val_loss = float('inf')
        self.wait = 0
        self.stopped_epoch = 0
        
    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get('val_loss')
        current_val_acc = logs.get('val_accuracy')
        
        # Check for high accuracy stopping condition
        if current_val_acc is not None and current_val_acc > self.accuracy_threshold:
            print(f'0.99990 accuracy reached - stopping training ###### ')
            self.model.stop_training = True
            self.stopped_epoch = epoch
            # Save model when stopping due to high accuracy
            self.model.save(f'{self.model_save_prefix}_epoch{epoch+1}.h5')
            return
            
        # Check for validation loss patience stopping
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f'Early stopping triggered after {self.patience} epochs without improvement ###### ')
                self.model.stop_training = True
                self.stopped_epoch = epoch
                # Save model when stopping due to patience
                self.model.save(f'{self.model_save_prefix}_epoch{epoch+1}.h5')
        
        # Save model after every epoch starting from epoch 1
        if epoch >= 1:
            self.model.save(f'{self.model_save_prefix}_epoch{epoch+1}.h5')

class LossHistory(keras.callbacks.Callback):
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

def main():   
    global SPF
    
    # Clear any previous TensorFlow session
    tf.keras.backend.clear_session()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--featureNumbers', type=int)
    args = parser.parse_args()
    if args.featureNumbers is not None:
        SPF = args.featureNumbers
    if SPF not in [3, 4, 8, 10]:
        return 0

    # Set random seeds for reproducibility
    tf.random.set_seed(39)
    np.random.seed(39)    

    # Read configuration parameters
    hiddenSize, learningRate = readConfigFiles()

    # Prepare data
    train_inputs, train_labels, input_size, valid_inputs, valid_labels = create_train_data()
    print(f'SP{SPF}F: input_size {input_size} ###### ')

    # Create model
    hidden_size = hiddenSize
    output_size = 2
    model = create_mlp_model(input_size, hidden_size, output_size)

    # Define loss function and optimizer
    # Note: from_logits=False because the final layer has LeakyReLU activation
    # For better classification performance, consider removing the final LeakyReLU 
    # and using from_logits=True
    criterion = losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = optimizers.AdamW(learning_rate=learningRate, weight_decay=1e-5)

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss=criterion,
        metrics=['accuracy']
    )

    # Setup callbacks
    early_stopping = CustomEarlyStopping(
        patience=5,
        accuracy_threshold=0.99990,
        model_save_prefix=f'mlp_SP{SPF}F'
    )
    
    loss_history = LossHistory(f'Loss_SP{SPF}F.txt')

    # Train the model using Keras fit method with custom callbacks
    print(f"Starting training with Keras built-in training loop and custom early stopping...")
    
    history = model.fit(
        x=train_inputs.astype(np.float32),
        y=train_labels.astype(np.int64),
        batch_size=5000,
        epochs=999,
        validation_data=(
            valid_inputs.astype(np.float32), 
            valid_labels.astype(np.int64)
        ),
        callbacks=[early_stopping, loss_history],
        verbose=1,
        shuffle=True  # Equivalent to shuffle=True in DataLoader
    )

    # Print final results
    final_epoch = len(history.history['loss'])
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    
    print(f'SP{SPF}F: Final Epoch [{final_epoch}/999], Train Loss: {final_train_loss:.6f}, Valid Loss: {final_val_loss:.6f}, Valid Acc: {final_val_acc:.6f} ###### ')
    print('Model training completed successfully! ###### ')
    
    return 1

if __name__ == "__main__":
    result = main()
    if result == 1:
        print('Normal finish!')
    else:
        print('Error happened!')