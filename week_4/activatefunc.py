import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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
    def __init__(self, activation_fn):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(28*28, 128)  # Input layer to hidden layer
        self.activation = activation_fn      # Use passed activation function
        self.output = nn.Linear(128, 10)     # Hidden layer to output layer
        self.softmax = nn.LogSoftmax(dim=1)  # Use LogSoftmax for numerical stability

    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden(x)
        x = self.activation(x)
        x = self.output(x)
        x = self.softmax(x)
        return x

# Create two models, one with ReLU and one with Sigmoid
model_relu = MLP(activation_fn=nn.ReLU())
model_sigmoid = MLP(activation_fn=nn.Sigmoid())

# 3. Model Compilation
def train_and_evaluate_model(model, epochs=10):
    criterion = nn.NLLLoss()  # Negative Log Likelihood Loss (used with LogSoftmax)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0
        correct = 0

        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)  # target is not one-hot encoded in PyTorch
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        train_losses.append(epoch_loss / len(train_loader))
        train_accuracy = 100. * correct / len(train_loader.dataset)
        train_accuracies.append(train_accuracy)

        # Validation
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader)
        test_accuracy = 100. * correct / len(test_loader.dataset)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

    return train_losses, test_losses, train_accuracies, test_accuracies

# Train both models
train_losses_relu, test_losses_relu, train_accuracies_relu, test_accuracies_relu = train_and_evaluate_model(model_relu)
train_losses_sigmoid, test_losses_sigmoid, train_accuracies_sigmoid, test_accuracies_sigmoid = train_and_evaluate_model(model_sigmoid)

# 4. Plotting the Results
plt.figure(figsize=(12, 5))

# Plot training losses
plt.subplot(1, 2, 1)
plt.plot(train_losses_relu, label='ReLU - Train Loss')
plt.plot(train_losses_sigmoid, label='Sigmoid - Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()

# Plot training accuracies
plt.subplot(1, 2, 2)
plt.plot(train_accuracies_relu, label='ReLU - Train Accuracy')
plt.plot(train_accuracies_sigmoid, label='Sigmoid - Train Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy Comparison')
plt.legend()

plt.tight_layout()
plt.show()

# Plot test losses and accuracies
plt.figure(figsize=(12, 5))

# Plot test losses
plt.subplot(1, 2, 1)
plt.plot(test_losses_relu, label='ReLU - Test Loss')
plt.plot(test_losses_sigmoid, label='Sigmoid - Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Test Loss Comparison')
plt.legend()

# Plot test accuracies
plt.subplot(1, 2, 2)
plt.plot(test_accuracies_relu, label='ReLU - Test Accuracy')
plt.plot(test_accuracies_sigmoid, label='Sigmoid - Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy Comparison')
plt.legend()

plt.tight_layout()
plt.show()
