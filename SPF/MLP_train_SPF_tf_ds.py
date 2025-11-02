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

def train_step(model, inputs, labels, criterion, optimizer):
    """Single training step"""
    with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        loss = criterion(labels, outputs)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def evaluate(model, dataset, criterion):
    """Evaluate model performance"""
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, labels in dataset:
        outputs = model(inputs, training=False)
        loss = criterion(labels, outputs)
        total_loss += loss.numpy()
        
        # Calculate accuracy
        predicted = tf.argmax(outputs, axis=1)
        correct += tf.reduce_sum(tf.cast(predicted == labels, tf.int32)).numpy()
        total += labels.shape[0]
    
    accuracy = correct / total
    return total_loss / len(dataset), accuracy

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
    criterion = losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = optimizers.AdamW(learning_rate=learningRate, weight_decay=1e-5)

    # Prepare TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_inputs.astype(np.float32), train_labels.astype(np.int64))
    ).batch(5000).shuffle(buffer_size=10000)
    
    valid_dataset = tf.data.Dataset.from_tensor_slices(
        (valid_inputs.astype(np.float32), valid_labels.astype(np.int64))
    ).batch(5000)

    # Training loop
    fp = open(f'Loss_SP{SPF}F.txt', 'w+')
    fp.write(f'Epoch, Train Loss, Valid Loss, Valid Acc\n')
    
    num_epochs = 999
    patience_max = 5
    patience_i = 0
    patience_diff = 0
    patience_diff_1 = 0
    valid_loss_old = 99999
    
    for epoch in range(num_epochs):
        # Training
        epoch_train_loss = 0
        for batch_inputs, batch_labels in train_dataset:
            batch_loss = train_step(model, batch_inputs, batch_labels, criterion, optimizer)
            epoch_train_loss += batch_loss.numpy()
        train_loss = epoch_train_loss / len(train_dataset)
        
        # Validation
        valid_loss, valid_acc = evaluate(model, valid_dataset, criterion)
        
        print(f'SP{SPF}F: Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.6f}, Valid Loss: {valid_loss:.6f}, Valid Acc: {valid_acc:.6f}  ###### ')
        fp.write(f' {epoch + 1}, {train_loss:.6f}, {valid_loss:.6f}, {valid_acc:.6f}\n')
        
        # Early stopping logic
        if valid_loss_old < valid_loss:
            patience_i += 1
            patience_diff += valid_loss - valid_loss_old
            patience_diff_1 = patience_diff / patience_i
        else:
            patience_diff += valid_loss - valid_loss_old
            if patience_diff < 0:
                patience_diff = 0
                patience_diff_1 = 0
                patience_i = 0
            else:
                patience_i = np.ceil(patience_diff / patience_diff_1)
                patience_diff_1 = patience_diff / patience_i

        # Check stopping conditions
        if valid_acc > 0.99990:
            print('0.99990 ###### ')
            model.save(f'mlp_SP{SPF}F_epoch{epoch+1}.h5')
            break
        if patience_i >= patience_max:
            print('patience_max ###### ')
            model.save(f'mlp_SP{SPF}F_epoch{epoch+1}.h5')
            break
        if epoch >= 1:
            model.save(f'mlp_SP{SPF}F_epoch{epoch+1}.h5')
        
        valid_loss_old = valid_loss

    fp.close()
    print('Model saved to mlp_model.h5 ###### ')
    return 1

if __name__ == "__main__":
    result = main()
    if result == 1:
        print('Normal finish!')
    else:
        print('Error happened!')