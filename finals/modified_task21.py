import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pickle
import matplotlib.pyplot as plt


# Load the dataset
with open('finals/data.pkl', 'rb') as file:
    data = pickle.load(file)

# Prepare the data
time = torch.tensor(data['time'], dtype=torch.float32).unsqueeze(1)  # Time as a single feature
goal_positions = torch.tensor(data['goal_positions'], dtype=torch.float32)  # Cartesian goal positions (x, y, z)
q_mes_all = torch.tensor(data['q_mes_all'], dtype=torch.float32)  # Joint positions for each time step

# Concatenate time and goal positions to create the input feature set
input_data = torch.cat((time.expand(-1, goal_positions.size(1)), goal_positions), dim=1)  # Shape: [num_samples, 4]
output_data = q_mes_all  # Target: joint positions for each time step, Shape: [num_samples, 7]

# Define your dataset
dataset = TensorDataset(input_data, output_data)

# Set sizes for training and testing
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

# Split the dataset without shuffling
train_dataset = TensorDataset(*dataset[:train_size])
test_dataset = TensorDataset(*dataset[train_size:])

# Create data loaders for training and testing sets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the MLP model for predicting joint positions
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Instantiate the model, loss function, and optimizer
input_size = input_data.shape[1]  # 4 (time + goal x, y, z)
hidden_size = 64  # Hidden layer size
output_size = q_mes_all.shape[1]  # 7 (joint positions)
model = MLP(input_size, hidden_size, output_size)

criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, criterion, optimizer, epochs=50):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

# Testing loop
def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
    print(f"Test Loss: {test_loss / len(test_loader):.4f}")


def plot_joint_trajectories(model, data_loader, title="Joint Trajectories", sample_rate=30):
    model.eval()
    actual_positions = []
    predicted_positions = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            actual_positions.extend(targets.numpy())
            predicted_positions.extend(outputs.numpy())
    
    # Convert to tensor for easier slicing
    actual_positions = torch.tensor(actual_positions)
    predicted_positions = torch.tensor(predicted_positions)
    
    # Sampling data for clearer visualization
    actual_positions = actual_positions[::sample_rate]
    predicted_positions = predicted_positions[::sample_rate]
    
    # Plot each joint's trajectory with larger figure size
    num_joints = actual_positions.shape[1]
    plt.figure(figsize=(12, 8))
    for i in range(num_joints):
        plt.subplot(3, 3, i + 1)
        plt.plot(actual_positions[:, i], label="Actual", linestyle='--')
        plt.plot(predicted_positions[:, i], label="Predicted")
        plt.title(f"Joint {i+1}")
        plt.xlabel("Sampled Time Step")
        plt.ylabel("Position")
        plt.legend()
    
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Train and test the model
train_model(model, train_loader, criterion, optimizer, epochs=50)
test_model(model, test_loader, criterion)

# Plot trajectories for training and testing sets with sampling
plot_joint_trajectories(model, train_loader, title="Training Set Joint Trajectories", sample_rate=30)
plot_joint_trajectories(model, test_loader, title="Testing Set Joint Trajectories", sample_rate=1)