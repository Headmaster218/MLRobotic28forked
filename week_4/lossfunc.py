import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Convert targets to one-hot encoding
def one_hot_encode(targets, num_classes=10):
    return torch.eye(num_classes)[targets]

# 1. Data Preparation
# Define transformations for the training and test sets
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to PyTorch Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize the dataset
])

# Download and load the training and test datasets
train_dataset = datasets.MNIST(root='./data', train=True,
                               transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False,
                              transform=transform, download=True)

# Create data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# 2. Model Construction
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(28*28, 128)  # Input layer to hidden layer
        self.relu = nn.ReLU()
        self.output = nn.Linear(128, 10)     # Hidden layer to output layer
        self.softmax = nn.LogSoftmax(dim=1)  # Use LogSoftmax for numerical stability

    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x

model = MLP()

# 3. Model Compilation
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Initialize lists to store losses for both methods
train_losses_mse = []
train_losses_nll = []

# Initialize lists to store accuracy for both methods
train_accuracies_mse = []
train_accuracies_nll = []

epochs = 10

# 4. Model Training for NLL Loss
criterion_nll = nn.NLLLoss()  # Negative Log Likelihood Loss (used with LogSoftmax)

# Train using NLLLoss
print("Training with NLLLoss:")
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    correct = 0

    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion_nll(nn.LogSoftmax(dim=1)(output), target)  # Apply log softmax and use NLLLoss
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    train_losses_nll.append(epoch_loss / len(train_loader))
    train_accuracy = 100. * correct / len(train_loader.dataset)
    train_accuracies_nll.append(train_accuracy)

    print(f'Epoch {epoch+1}/{epochs}, Loss: {train_losses_nll[-1]:.4f}, Accuracy: {train_accuracy:.2f}%')

# Re-initialize model and optimizer for fair comparison
model = MLP()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 5. Model Training for MSE Loss
criterion_mse = nn.MSELoss()  # Mean Squared Error Loss

print("\nTraining with MSELoss:")
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    correct = 0

    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        target_one_hot = one_hot_encode(target)  # Convert target to one-hot encoding
        loss = criterion_mse(torch.exp(nn.LogSoftmax(dim=1)(output)), target_one_hot)  # Apply softmax before MSE
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    train_losses_mse.append(epoch_loss / len(train_loader))
    train_accuracy = 100. * correct / len(train_loader.dataset)
    train_accuracies_mse.append(train_accuracy)

    print(f'Epoch {epoch+1}/{epochs}, Loss: {train_losses_mse[-1]:.4f}, Accuracy: {train_accuracy:.2f}%')

# 6. Plot Comparison
plt.figure(figsize=(12, 5))

# Plot loss comparison
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), train_losses_nll, label="NLL Loss")
plt.plot(range(1, epochs+1), train_losses_mse, label="MSE Loss")
plt.title("Training Loss Comparison")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Plot accuracy comparison
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), train_accuracies_nll, label="NLL Accuracy")
plt.plot(range(1, epochs+1), train_accuracies_mse, label="MSE Accuracy")
plt.title("Training Accuracy Comparison")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.show()

