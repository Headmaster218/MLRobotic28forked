import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
m = 1.0  # Mass (kg)
b = 10  # Friction coefficient
k_p = 50  # Proportional gain
k_d = 10  # Derivative gain
dt = 0.01  # Time step
num_samples = 1000  # Number of samples in dataset

# Generate synthetic data for trajectory tracking
t = np.linspace(0, 10, num_samples)
q_target = np.sin(t)
dot_q_target = np.cos(t)

# Initial conditions for training data generation
q = 0
dot_q = 0
X = []
Y = []

for i in range(num_samples):
    tau = k_p * (q_target[i] - q) + k_d * (dot_q_target[i] - dot_q)
    ddot_q_real = (tau - b * dot_q) / m
    ddot_q_ideal = tau / m
    ddot_q_error = ddot_q_ideal - ddot_q_real
    X.append([q, dot_q, q_target[i], dot_q_target[i]])
    Y.append(ddot_q_error)
    dot_q += ddot_q_real * dt
    q += dot_q * dt

X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32).view(-1, 1)
train_dataset = TensorDataset(X_tensor, Y_tensor)

# Define MLP model classes
class DeepCorrectorMLP(nn.Module):
    def __init__(self, num_hidden_units=32):
        super(DeepCorrectorMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, num_hidden_units),
            nn.ReLU(),
            nn.Linear(num_hidden_units, num_hidden_units),
            nn.ReLU(),
            nn.Linear(num_hidden_units, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

class ShallowCorrectorMLP(nn.Module):
    def __init__(self, num_hidden_units=32):
        super(ShallowCorrectorMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, num_hidden_units),
            nn.ReLU(),
            nn.Linear(num_hidden_units, 1)
        )

    def forward(self, x):
        return self.layers(x)

# Parameter settings for testing
hidden_units_list = [32, 64, 96, 128]
learning_rates = [1e-1, 1e-2, 1e-3, 1e-4]
batch_sizes = [4,8,16,32, 64, 128, 256, 1000]
epochs = 50

# Automated Training and Evaluation for Both Models
for batch_size in batch_sizes:
    results_deep = np.zeros((len(hidden_units_list), len(learning_rates)))
    results_shallow = np.zeros((len(hidden_units_list), len(learning_rates)))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for i, hidden_units in enumerate(hidden_units_list):
        for j, lr in enumerate(learning_rates):
            # DeepCorrectorMLP
            model_deep = DeepCorrectorMLP(num_hidden_units=hidden_units)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model_deep.parameters(), lr=lr)
            
            train_losses_deep = []
            model_deep.train()
            for epoch in range(epochs):
                epoch_loss = 0
                for data, target in train_loader:
                    optimizer.zero_grad()
                    output = model_deep(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                train_losses_deep.append(epoch_loss / len(train_loader))
            
            results_deep[i, j] = train_losses_deep[-1]
            
            # ShallowCorrectorMLP
            model_shallow = ShallowCorrectorMLP(num_hidden_units=hidden_units)
            optimizer = optim.Adam(model_shallow.parameters(), lr=lr)
            
            train_losses_shallow = []
            model_shallow.train()
            for epoch in range(epochs):
                epoch_loss = 0
                for data, target in train_loader:
                    optimizer.zero_grad()
                    output = model_shallow(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                train_losses_shallow.append(epoch_loss / len(train_loader))
            
            results_shallow[i, j] = train_losses_shallow[-1]
            
            print(f'Batch Size: {batch_size}, Hidden Units: {hidden_units}, LR: {lr}, Deep MLP Loss: {train_losses_deep[-1]:.6f}, Shallow MLP Loss: {train_losses_shallow[-1]:.6f}')

    # Apply log transformation for brightness coloring only
    results_deep_log = np.log10(np.clip(results_deep, 1e-6, None))
    results_shallow_log = np.log10(np.clip(results_shallow, 1e-6, None))
    
    # Plot side-by-side heatmaps for Deep and Shallow MLP models
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # DeepCorrectorMLP Heatmap with brightness and green annotation
    sns.heatmap(results_deep_log, annot=results_deep, fmt=".4f", xticklabels=learning_rates, yticklabels=hidden_units_list,
                cmap="Greys", ax=ax1, annot_kws={"color": "green"})
    ax1.set_xlabel("Learning Rate")
    ax1.set_ylabel("Hidden Units")
    ax1.set_title(f"Training Loss for DeepCorrectorMLP (Log Brightness), Batch Size {batch_size}")

    # ShallowCorrectorMLP Heatmap with brightness and green annotation
    sns.heatmap(results_shallow_log, annot=results_shallow, fmt=".4f", xticklabels=learning_rates, yticklabels=hidden_units_list,
                cmap="Greys", ax=ax2, annot_kws={"color": "green"})
    ax2.set_xlabel("Learning Rate")
    ax2.set_ylabel("Hidden Units")
    ax2.set_title(f"Training Loss for ShallowCorrectorMLP (Log Brightness), Batch Size {batch_size}")
    
    plt.tight_layout()
    plt.show()
