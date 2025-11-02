import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset
import glob
import os
from pathlib import Path
import argparse
import gc

SPF = 10#3,4,8,10

def readConfigFiles():
    # The directory where the current script is located
    global SPF
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent

    file_path = parent_dir / 'hiddenSize.txt'
    with file_path.open('r', encoding='utf-8') as f:
        content = f.read()
    lines = content.split('\n')
    if(SPF==3):
        hidden_size=int(lines[1])
    if(SPF==4):
        hidden_size=int(lines[3])
    if(SPF==8):
        hidden_size=int(lines[5])
    if(SPF==10):
        hidden_size=int(lines[7])

    file_path = parent_dir / 'chosenPthFile.txt'
    with file_path.open('r', encoding='utf-8') as f:
        content = f.read()
    lines = content.split('\n')
    if(SPF==3):
        pthFileName=lines[1]
    if(SPF==4):
        pthFileName=lines[3]
    if(SPF==8):
        pthFileName=lines[5]
    if(SPF==10):
        pthFileName=lines[7]        

    pthFileName=parent_dir /'savedPthFiles_new'/ pthFileName

    return hidden_size,pthFileName

# Reading point cloud data from a txt file
def read_points_from_txt(file_path):
    # Read a txt file using numpy, assuming the data is comma-separated
    points = np.loadtxt(file_path, delimiter=',')
    return points

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, int(hidden_size/2))
        self.fc2 = nn.Linear(int(hidden_size/2), hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.fc7 = nn.Linear(hidden_size, hidden_size)
        self.fc8 = nn.Linear(hidden_size, hidden_size)
        self.fc9 = nn.Linear(hidden_size, int(hidden_size/2))        
        self.fc10 = nn.Linear(int(hidden_size/2), 6)   
        self.fc11 = nn.Linear(6, output_size)

    def forward(self, x):
        alpha=0.01
        x = F.leaky_relu(self.fc1(x),negative_slope=alpha)
        x = F.leaky_relu(self.fc2(x),negative_slope=alpha)
        x = F.leaky_relu(self.fc3(x),negative_slope=alpha)
        x = F.leaky_relu(self.fc4(x),negative_slope=alpha)
        x = F.leaky_relu(self.fc5(x),negative_slope=alpha)
        x = F.leaky_relu(self.fc6(x),negative_slope=alpha)
        x = F.leaky_relu(self.fc7(x),negative_slope=alpha)
        x = F.leaky_relu(self.fc8(x),negative_slope=alpha)
        x = F.leaky_relu(self.fc9(x),negative_slope=alpha)
        x = F.leaky_relu(self.fc10(x),negative_slope=alpha)
        x = F.leaky_relu(self.fc11(x),negative_slope=alpha)
        return x

# Creating Data
def create_test_data(file_path):
    global SPF

    quality_Feature=0
    # Reading point cloud data
    points = read_points_from_txt(file_path)
    if(np.size(points,1)==16):
        SPF=SPF+2
    print(f'SP{SPF}F: read_points_from_txt ###### ')

    if(np.size(points,1)==14):
        if(SPF==3):
            input_data_2d=points[:,[4,6,8]]
        if(SPF==4):
            input_data_2d=points[:,[3,4,6,8]]
        if(SPF==8):
            input_data_2d=points[:,[3,4,5,6,8,9,10,11]]
        if(SPF==10):
            input_data_2d=points[:,3:13]
    if(np.size(points,1)==16):
        quality_Feature=1
        if(SPF==5):
            input_data_2d=points[:,[4,6,8,13,14]]
        if(SPF==6):
            input_data_2d=points[:,[3,4,6,8,13,14]]
        if(SPF==10):
            input_data_2d=points[:,[3,4,5,6,8,9,10,11,13,14]]
        if(SPF==12):
            input_data_2d=points[:,3:15]
    print(f'SP{SPF}F: input_data prepared ###### ')

    # Label Data
    labels = points[:, 0]

    return input_data_2d, labels,input_data_2d.shape[1],quality_Feature

# Validation Function
def evaluate(model, criterion, valid_loader, device):
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0
    pred_data = []
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pred_data.append(predicted)
    accuracy = correct / total            
    return epoch_loss / len(valid_loader), accuracy, pred_data


def main():   

    global SPF
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('--featureNumbers', type=int)
    args = parser.parse_args()
    if args.featureNumbers!=None:
        SPF = args.featureNumbers
    if (SPF!=3 and SPF!=4 and SPF!=8 and SPF!=10):
        return 0

    # Set the random seed so that the results are reproducible
    torch.manual_seed(39)
    np.random.seed(39)
    # Free up all unused memory on the GPU
    torch.cuda.empty_cache()

    hiddenSize, pthFileName = readConfigFiles()

    # Data preparation
    data_path = Path(__file__).parent
    folder_names = glob.glob(os.path.join(data_path, '[0-9][0-9][0-9]_[0-9][0-9][0-9]_[0-9][0-9][0-9]'))

    for iF in range(0,len(folder_names)):
        file_path = os.path.join(folder_names[iF], 'SPF_allClass_testData.txt')
        test_inputs, test_labels,input_size,quality_Feature = create_test_data(file_path)

        # Creating a Model Instance
        hidden_size = hiddenSize # Hidden layer size
        output_size=2
        model = MLP(input_size, hidden_size,output_size)

        # Defining loss function and optimizer
        criterion = nn.CrossEntropyLoss()

        # Move data to GPU (if available)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        test_inputs, test_labels = test_inputs, test_labels.astype(np.int64)

        # Convert to TensorDataset
        test_dataset = TensorDataset(torch.from_numpy(test_inputs).to(torch.float32).to(device), \
                                    torch.from_numpy(test_labels).to(torch.long).to(device))
        # Creating a DataLoader
        test_loader = DataLoader(test_dataset, batch_size=5000, shuffle=False)

        # Loading model weights
        model.load_state_dict(torch.load(pthFileName))
        print(f'SP{SPF}F: Model loaded from mlp_model.pth ###### ')
        
        loss, preAcc, pred_Value=evaluate(model, criterion, test_loader, device)
        print(f' SP{SPF}F: Test Loss: {loss:.4f}, Test Acc: {preAcc:.4f} ###### ')
        #
        del test_loader,test_inputs,test_labels,test_dataset
        # Manually collect garbage to release unreferenced objects
        gc.collect()
        fp=open(folder_names[iF]+f'\\SP{SPF}F_allClass_testData_Prediction.txt','w+')
        
        # Concatenate a list of tensors into a single tensor
        pred_data_tensor = torch.cat(pred_Value)
        # Move tensor to CPU and convert to NumPy array
        pred_data_numpy = pred_data_tensor.cpu().numpy()
        # pred_data_str=''
        # for tmpNum in pred_data_numpy:
        #     pred_data_str=pred_data_str+str(tmpNum)+',\n'
        # Convert a NumPy array to a string, separated by commas and newlines
        pred_data_str = ',\n'.join(map(str, pred_data_numpy))
        fp.write(pred_data_str)
        fp.close()
        print(folder_names[iF]+f' SP{SPF}F: Prediction finished!')
        if(quality_Feature==1):
            SPF=SPF-2
    
    return 1

if __name__ == "__main__":
    result = main()
    if(result==1):
        print('Normal finish!')
    else:
        print('Error happened!')

