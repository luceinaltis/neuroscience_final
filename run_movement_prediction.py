import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

def calculate_rmse(predictions, targets):
    return torch.sqrt(torch.mean((predictions - targets) ** 2))

max_time = 0   
spikes = []
handPos = []
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
        handPos.append(np.array(json_data[idx][trial]['handPos']))
    directions.extend([idx for i in range(100)])
    
print('about spikes')
for idx in range(len(spikes)):
    new_shape = (98, max_time)
    new_spike = np.zeros(new_shape, dtype=spikes[idx].dtype)
    new_spike[0:spikes[idx].shape[0], 0:spikes[idx].shape[1]] = spikes[idx]
    spikes[idx] = new_spike.transpose(1, 0) # Reshape to (batch_size, 975, 98)
    
print('about handPos')
for idx in range(len(handPos)):
    new_shape = (max_time, 3) # (time, dir)
    handPos[idx] = np.array(handPos[idx]).transpose(1, 0)
    handPos[idx] = handPos[idx][1:len(handPos[idx])][:] - handPos[idx][:-1][:]
    new_handPos = np.zeros(new_shape, dtype=handPos[idx].dtype)
    new_handPos[0:handPos[idx].shape[0], 0:handPos[idx].shape[1]] = handPos[idx]
    handPos[idx] = new_handPos


# Normalize your data (this step depends on your data's nature)
# spikes_normalized = ...

# Convert data to PyTorch tensors
spikes_tensor = torch.Tensor(np.array(spikes))
handVec_tensor = torch.Tensor(np.array(handPos)) # (time, dir)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(spikes_tensor, handVec_tensor, test_size=0.2, random_state=42)

print('create dataloader')
# Create DataLoader for both training and testing
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256, shuffle=True)


# Define the RNN model
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.hidden_size = 600
        self.lstm = nn.LSTM(input_size=98, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_size, 3)  # Output is 3D position for each time step
        

    def forward(self, x):
        x, (hn, cn) = self.lstm(x)
        
        #hn = hn.view(-1, self.hidden_size) # Reshaping the data for starting LSTM network
        #x = self.relu(x) #pre-processing for first layer
        x = self.fc1(x)  # Using the last output of the sequence
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model, define loss function and optimizer
for k in range(1):
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    #optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1)

    # Training the model and displaying accuracy
    for epoch in range(5):  # Number of epochs
        total_rmse = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            rmse = calculate_rmse(output, target)
            total_rmse += rmse.item()

        average_rmse = total_rmse / len(train_loader)
        print(f'epoch {epoch+1}, Average RMSE on train data: {average_rmse}')

    # Evaluate the model
    model.eval()
    total_rmse = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data.to(device))
            rmse = calculate_rmse(outputs, target)
            total_rmse += rmse.item()
            
    average_rmse = total_rmse / len(test_loader)
    print(f'Average RMSE on test data: {average_rmse}')



from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

with torch.no_grad():
    for data, target in test_loader:
        outputs = model(data.to(device))
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.view_init(elev=30, azim=30, roll=0)

        for i in range(len(target)):
            x = torch.cumsum(target[i][:, 0], dim=0)
            y = torch.cumsum(target[i][:, 1], dim=0)
            z = torch.cumsum(target[i][:, 2], dim=0)
            ax.plot3D(x, y, z, color="red")
            
            x = torch.cumsum(outputs[i][:, 0], dim=0)
            y = torch.cumsum(outputs[i][:, 1], dim=0)
            z = torch.cumsum(outputs[i][:, 2], dim=0)
            ax.plot3D(x, y, z, color="blue")
            
        plt.show()
