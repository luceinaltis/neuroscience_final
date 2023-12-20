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
        
        if np.array(json_data[idx][trial]['spikes']).shape[1] > max_time:
            max_time = np.array(json_data[idx][trial]['spikes']).shape[1]
    directions.extend([idx for i in range(100)])

for idx in range(len(spikes)):
    new_shape = (98, max_time)
    new_spike = np.zeros(new_shape, dtype=spikes[idx].dtype)
    new_spike[0:spikes[idx].shape[0], 0:spikes[idx].shape[1]] = spikes[idx]
    spikes[idx] = new_spike


# Normalize your data (this step depends on your data's nature)
# spikes_normalized = ...

# Convert data to PyTorch tensors



spikes_tensor = torch.Tensor(np.array(spikes))
directions_tensor = torch.Tensor(np.array(directions)).long()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(spikes_tensor, directions_tensor, test_size=0.2, random_state=42)

print('create dataloader')
# Create DataLoader for both training and testing
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256, shuffle=True)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epoch_arr = []
loss_arr = []
acc_arr = [] 

# Instantiate the model, define loss function and optimizer
for k in range(1):
    model = CNN2()
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1)
    step = 0
    model.to(device)

    # Training the model and displaying accuracy
    for epoch in range(60):  # Number of epochs
        running_loss = 0.0
        correct = 0
        total = 0

        for data, target in train_loader:
            data
            optimizer.zero_grad()
            
            output = model(data.to(device))
            loss = criterion(output, target.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f'epoch {epoch+1}, loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        
        epoch_arr.append(epoch+1)
        loss_arr.append(epoch_loss)
        acc_arr.append(epoch_acc)

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target.to(device)).sum().item()

    print(f'Test Accuracy: {100 * correct / total}%')

import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()

color = 'tab:green'
ax1.plot(epoch_arr, loss_arr, color=color)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color=color)
ax1.tick_params(axis='y', labelcolor=color)

color = 'tab:blue'
ax2 = ax1.twinx()
ax2.plot(epoch_arr, acc_arr, color=color)
ax2.set_ylabel('Accuracy (%)', color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 100)

fig.tight_layout()
plt.show()

breakpoint()