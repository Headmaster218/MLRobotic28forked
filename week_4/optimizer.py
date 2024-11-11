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
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

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
        x = self.softmax(x)
        return x

# Function for training and evaluation
def train_and_evaluate(optimizer_name, optimizer, epochs=10):
    model = MLP()
    criterion = nn.NLLLoss()

    # Initialize optimizer (passed as argument)
    optimizer = optimizer(model.parameters(), lr=0.01)

    train_losses = []
    test_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0

        # Training loop
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        train_losses.append(epoch_loss / len(train_loader))
        train_accuracy = 100. * correct / len(train_loader.dataset)

        # Evaluate on the test set
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

        print(f'Optimizer: {optimizer_name}, Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {train_losses[-1]:.4f}, '
              f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

    return train_losses, test_losses, test_accuracies

# 3. Optimizer Comparison
optimizers = {
    'SGD': optim.SGD,
    'Adam': optim.Adam,
    'RMSprop': optim.RMSprop
}

results = {}

for name, optimizer in optimizers.items():
    train_losses, test_losses, test_accuracies = train_and_evaluate(name, optimizer)
    results[name] = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies
    }

# 4. Plotting Results
epochs = range(1, 11)

# Plot Training Losses
plt.figure(figsize=(10, 5))
for name in optimizers.keys():
    plt.plot(epochs, results[name]['train_losses'], label=f'Train Loss - {name}')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.show()

# Plot Test Losses
plt.figure(figsize=(10, 5))
for name in optimizers.keys():
    plt.plot(epochs, results[name]['test_losses'], label=f'Test Loss - {name}')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Test Loss Comparison')
plt.legend()
plt.show()

# Plot Test Accuracies
plt.figure(figsize=(10, 5))
for name in optimizers.keys():
    plt.plot(epochs, results[name]['test_accuracies'], label=f'Test Accuracy - {name}')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy Comparison')
plt.legend()
plt.show()
