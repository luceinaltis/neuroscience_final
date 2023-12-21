import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

max_time = 0   
spikes = []
directions = []
traj = []
spikes_org = []
traj_org = []

print('open data')
# Open data with json format
with open('monkeydata.json') as json_file:
    json_data = json.load(json_file)
    json_data = json_data['trial']

print('preprocess data')
# Preprocess your data
# Assuming 'spikes' is a list of neural spike data and 'direction' is the target
for idx in range(8):
    for trial in range(100):
        spikes.append(np.array(json_data[idx][trial]['spikes']))
        spikes_org.append(torch.tensor(np.array(json_data[idx][trial]['spikes'])))
        traj.append((np.array(json_data[idx][trial]['handPos'])))
        traj_org.append(torch.tensor(np.array(json_data[idx][trial]['handPos'])))
        

        if np.array(json_data[idx][trial]['spikes']).shape[1] > max_time:
            max_time = np.array(json_data[idx][trial]['spikes']).shape[1]


        

    directions.extend([idx for i in range(100)])



for idx in range(len(spikes)):
    new_shape = (98, max_time)
    new_spike = np.zeros(new_shape, dtype=spikes[idx].dtype)
    new_spike[0:spikes[idx].shape[0], 0:spikes[idx].shape[1]] = spikes[idx]
    spikes[idx] = new_spike

    new_shape = (3, max_time)
    new_traj = np.zeros(new_shape, dtype=traj[idx].dtype)
    new_traj[0:traj[idx].shape[0], 0:traj[idx].shape[1]] = traj[idx]
    traj[idx] = new_traj

 

# Normalize your data (this step depends on your data's nature)
# spikes_normalized = ...

# Convert data to PyTorch tensors



spikes_tensor = torch.Tensor(np.array(spikes))
directions_tensor = torch.Tensor(np.array(directions)).long()
traj_tensor = torch.Tensor(np.array(traj))


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(spikes_tensor, directions_tensor, test_size=0.2, random_state=42)

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(spikes_tensor, traj_tensor, test_size=0.2, random_state=42)



print('create dataloader')
# Create DataLoader for both training and testing
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256, shuffle=True)

train_data_2 = TensorDataset(X_train_2, y_train_2)
test_data_2 = TensorDataset(X_test_2, y_test_2)
train_loader_2 = DataLoader(train_data_2, batch_size=256, shuffle=True)
test_loader_2 = DataLoader(test_data_2, batch_size=256, shuffle=True)

# Define the CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=98, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        # The size here is an example, adjust according to your model's architecture
        self.fc1 = nn.Linear(32 * 486, 64)  # Adjust the input features
        self.fc2 = nn.Linear(64, 8)

    def forward(self, x):
        
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.flatten(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)

        if self.training == False:
            x = torch.softmax(x, dim=1)
        
        return x
    
class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=98, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        # The size here is an example, adjust according to your model's architecture
        self.fc1 = nn.Linear(32 * 242, 64)  # Adjust the input features
        self.fc2 = nn.Linear(64, 8)

    def forward(self, x):
        
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.flatten(x)

        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)

        if self.training == False:
            x = torch.softmax(x, dim=1)
        
        return x
    

class CBR_1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=9, stride=1, padding=4):
        super().__init__()
        self.seq_list = [
            nn.Conv1d(in_channels, out_channels, kernel, stride, padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        ]
        self.seq = nn.Sequential(*self.seq_list)

    def forward(self, x):
        return self.seq(x)

class Unet_1D(nn.Module):
    def __init__(self, class_n, layer_n):
        super().__init__()

        ### ------- encoder -----------
        self.enc1_1 = CBR_1D(98, layer_n)
        self.enc1_2 = CBR_1D(layer_n, layer_n)
        self.enc1_3 = CBR_1D(layer_n, layer_n)

        self.enc2_1 = CBR_1D(layer_n, layer_n * 2)
        self.enc2_2 = CBR_1D(layer_n * 2, layer_n * 2)

        self.enc3_1 = CBR_1D(layer_n * 2, layer_n * 4)
        self.enc3_2 = CBR_1D(layer_n * 4, layer_n * 4)

        self.enc4_1 = CBR_1D(layer_n * 4, layer_n * 8)
        self.enc4_2 = CBR_1D(layer_n * 8, layer_n * 8)

        ### ------- decoder -----------
        self.upsample_3 = nn.ConvTranspose1d(layer_n * 8, layer_n * 8, kernel_size=8, stride=2, padding=3)
        self.dec3_1 = CBR_1D(layer_n * 4 + layer_n * 8, layer_n * 4)
        self.dec3_2 = CBR_1D(layer_n * 4, layer_n * 4)

        self.upsample_2 = nn.ConvTranspose1d(layer_n * 4, layer_n * 4, kernel_size=8, stride=2, padding=3)
        self.dec2_1 = CBR_1D(layer_n * 2 + layer_n * 4, layer_n * 2)
        self.dec2_2 = CBR_1D(layer_n * 2, layer_n * 2)

        self.upsample_1 = nn.ConvTranspose1d(layer_n * 2, layer_n * 2, kernel_size=8, stride=2, padding=3)
        self.dec1_1 = CBR_1D(layer_n * 1 + layer_n * 2, layer_n * 1)
        self.dec1_2 = CBR_1D(layer_n * 1, layer_n * 1)
        self.dec1_3 = CBR_1D(layer_n * 1, class_n)  # Adjusted the output channels
        self.dec1_4 = CBR_1D(class_n, class_n)

    def forward(self, x):
        enc1 = self.enc1_1(x)
        enc1 = self.enc1_2(enc1)
        enc1 = self.enc1_3(enc1)

        enc2 = nn.functional.max_pool1d(enc1, 2)
        enc2 = self.enc2_1(enc2)
        enc2 = self.enc2_2(enc2)

        enc3 = nn.functional.max_pool1d(enc2, 2)
        enc3 = self.enc3_1(enc3)
        enc3 = self.enc3_2(enc3)

        enc4 = nn.functional.max_pool1d(enc3, 2)
        enc4 = self.enc4_1(enc4)
        enc4 = self.enc4_2(enc4)

        dec3 = self.upsample_3(enc4)
        dec3 = nn.functional.interpolate(dec3, size=enc3.size()[-1], mode='linear', align_corners=True)
        
        dec3 = self.dec3_1(torch.cat([enc3, dec3], dim=1))  ##concat
        dec3 = self.dec3_2(dec3)

        dec2 = self.upsample_2(dec3)
        dec2 = nn.functional.interpolate(dec2, size=enc2.size()[-1], mode='linear', align_corners=True)
        
        dec2 = self.dec2_1(torch.cat([enc2, dec2], dim=1))  ##concat
        dec2 = self.dec2_2(dec2)

        dec1 = self.upsample_1(dec2)
        dec1 = nn.functional.interpolate(dec1, size=enc1.size()[-1], mode='linear', align_corners=True)


        dec1 = self.dec1_1(torch.cat([enc1, dec1], dim=1))  ##concat
        dec1 = self.dec1_2(dec1)
        dec1 = self.dec1_3(dec1)
        out = self.dec1_4(dec1)

        return out
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model, define loss function and optimizer
for k in range(1):
    #model = CNN2()
    model = Unet_1D(class_n = 3, layer_n = 32)

    #criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    #optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1)
    step = 0
    model.to(device)

    # Training the model and displaying accuracy
    # for epoch in range(60):  # Number of epochs
    #     running_loss = 0.0
    #     correct = 0
    #     total = 0

    #     for data, target in train_loader:
    #         data
    #         optimizer.zero_grad()
            
    #         output = model(data.to(device))
    #         loss = criterion(output, target.to(device))
    #         loss.backward()
    #         optimizer.step()

    #         running_loss += loss.item()
    #         _, predicted = torch.max(output.data, 1)
    #         total += target.size(0)
    #         correct += (predicted == target).sum().item()

    #     epoch_loss = running_loss / len(train_loader)
    #     epoch_acc = 100 * correct / total
    #     print(f'epoch {epoch+1}, loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

    for epoch in range(60):  # Number of epochs
        running_loss = 0.0
        correct = 0
        total = 0

        for data, target in train_loader_2:

            optimizer.zero_grad()
            
            output = model(data.to(device))
            loss = criterion(output, target.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()


        epoch_loss = running_loss / len(train_loader_2)
        epoch_acc = 0
        print(f'epoch {epoch+1}, loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader_2:
            outputs = model(data.to(device))
            mse = criterion(output, target.to(device))
            

    print(f'Test Accuracy: {mse}%')



breakpoint()