import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import argparse

SPF = 10 #3,4,8,10

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

    file_path = parent_dir / 'learningRate.txt'
    with file_path.open('r', encoding='utf-8') as f:
        content = f.read()
    lines = content.split('\n')
    if(SPF==3):
        learningRate=float(lines[1])
    if(SPF==4):
        learningRate=float(lines[3])
    if(SPF==8):
        learningRate=float(lines[5])
    if(SPF==10):
        learningRate=float(lines[7])        

    return hidden_size,learningRate

# Read point cloud data from txt file
def read_points_from_txt(file_path):
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

# Creating training data
def create_train_data():
    # Specify the txt file path
    global SPF
    current_dir = Path(__file__).parent
    file_path = current_dir / 'SPF_Land_Water_Training.txt'    

    # Read point cloud data
    points = read_points_from_txt(file_path)
    if(np.size(points,1)==15):
        SPF=SPF+2
    print(f'SP{SPF}F: read_points_from_txt ###### ')

    if(np.size(points,1)==15):
        if(SPF==5):
            input_data_2d=points[:,[4,6,8,13,14]]
        if(SPF==6):
            input_data_2d=points[:,[3,4,6,8,13,14]]
        if(SPF==10):
            input_data_2d=points[:,[3,4,5,6,8,9,10,11,13,14]]
        if(SPF==12):
            input_data_2d=points[:,3:]
    if(np.size(points,1)==13):
        if(SPF==3):
            input_data_2d=points[:,[4,6,8]]
        if(SPF==4):
            input_data_2d=points[:,[3,4,6,8]]
        if(SPF==8):
            input_data_2d=points[:,[3,4,5,6,8,9,10,11]]
        if(SPF==10):
            input_data_2d=points[:,3:]

    train_size=int(input_data_2d.shape[0]*0.8)

    train_data=input_data_2d[1:train_size,:] 
    valid_data=input_data_2d[train_size:,:] 
    print(f'SP{SPF}F: input_data prepared ###### ')

    # Simulating label data
    train_labels = points[1:train_size, 0]
    valid_labels = points[train_size:, 0]

    return train_data, train_labels,input_data_2d.shape[1],valid_data,valid_labels

# Training function
def train(model, optimizer, criterion, train_loader, device):
    model.train()
    epoch_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(train_loader)


def evaluate(model, criterion, valid_loader, device):
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return epoch_loss / len(valid_loader), accuracy


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

    # set random seed
    torch.manual_seed(39)
    np.random.seed(39)    

    hiddenSize, learningRate=readConfigFiles()

    # prepare data
    train_inputs, train_labels,input_size,valid_inputs,valid_labels = create_train_data()
    print(f'SP{SPF}F: input_size {input_size} ###### ')

    # create model
    #hidden_size = 512
    hidden_size=hiddenSize
    output_size=2
    model = MLP(input_size, hidden_size,output_size)

    # set loss function and Optimizer
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.AdamW(model.parameters(), lr=learningRate)  # Adam
    optimizer = optim.AdamW(model.parameters(), lr=learningRate, weight_decay = 1e-5)

    # use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_inputs, train_labels = train_inputs, train_labels.astype(np.int64)
    valid_inputs, valid_labels = valid_inputs, valid_labels.astype(np.int64)

    # transfer to TensorDataset
    train_dataset = TensorDataset(torch.from_numpy(train_inputs).to(torch.float32).to(device), \
                                  torch.from_numpy(train_labels).to(torch.long).to(device))
    valid_dataset = TensorDataset(torch.from_numpy(valid_inputs).to(torch.float32).to(device), \
                                  torch.from_numpy(valid_labels).to(torch.long).to(device))
    # create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=5000, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=5000, shuffle=True)

    fp=open(f'Loss_SP{SPF}F.txt','w+')
    fp.write(f'Epoch, Train Loss, Valid Loss, Valid Acc\n')
    #
    num_epochs = 999
    #patience max
    patience_max =5
    patience_i=0
    patience_diff=0
    patience_diff_1=0
    valid_loss_old=99999
    #
    for epoch in range(num_epochs):
        train_loss = train(model, optimizer, criterion, train_loader, device)
        valid_loss, valid_acc = evaluate(model, criterion, valid_loader, device)
        print(f'SP{SPF}F: Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.6f}, Valid Loss: {valid_loss:.6f}, Valid Acc: {valid_acc:.6f}  ###### ')
        fp.write(f' {epoch + 1}, {train_loss:.6f}, {valid_loss:.6f}, {valid_acc:.6f}\n')
        if(valid_loss_old<valid_loss):
            patience_i=patience_i+1
            patience_diff=patience_diff+valid_loss-valid_loss_old
            patience_diff_1=patience_diff/patience_i
        else:
            patience_diff=patience_diff+valid_loss-valid_loss_old
            if(patience_diff<0):
                patience_diff=0
                patience_diff_1=0
                patience_i=0
            else:
                patience_i=np.ceil(patience_diff/patience_diff_1)
                patience_diff_1=patience_diff/patience_i

        if(valid_acc>0.99990):
            print('0.99990 ###### ')
            torch.save(model.state_dict(), f'mlp_SP{SPF}F_epoch{epoch+1}.pth')
            break
        if(patience_i>=patience_max):
            print('patience_max ###### ')
            torch.save(model.state_dict(), f'mlp_SP{SPF}F_epoch{epoch+1}.pth')
            break
        if(epoch>=1):
            torch.save(model.state_dict(), f'mlp_SP{SPF}F_epoch{epoch+1}.pth')
        #
        valid_loss_old=valid_loss

    fp.close()
    print('Model saved to mlp_model.pth ###### ')
    return 1


if __name__ == "__main__":
    result = main()
    if(result==1):
        print('Normal finish!')
    else:
        print('Error happened!')