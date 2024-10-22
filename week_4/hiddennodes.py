import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. Data Preparation
# Define transformations for the training and test sets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
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
    def __init__(self, hidden_nodes):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(28*28, hidden_nodes)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_nodes, 10)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)
        return x

# 3. Training and Evaluation Function
def train_and_evaluate(hidden_nodes):
    model = MLP(hidden_nodes)
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    epochs = 10
    train_losses = []
    valid_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        
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

    return train_losses, test_loss, test_accuracy

# 4. Running the experiments with different hidden node sizes
hidden_node_sizes = [64, 128, 256]
results = {}

for hidden_nodes in hidden_node_sizes:
    print(f'\nTraining with {hidden_nodes} hidden nodes:')
    train_losses, test_loss, test_accuracy = train_and_evaluate(hidden_nodes)
    results[hidden_nodes] = (train_losses, test_loss, test_accuracy)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

# 5. Plotting the results
plt.figure(figsize=(12, 6))

# Plot training losses
for hidden_nodes in hidden_node_sizes:
    train_losses, test_loss, test_accuracy = results[hidden_nodes]
    plt.plot(train_losses, label=f'{hidden_nodes} hidden nodes')

plt.title('Training Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Display final test accuracies for different hidden node sizes
for hidden_nodes in hidden_node_sizes:
    _, test_loss, test_accuracy = results[hidden_nodes]
    print(f'Hidden Nodes: {hidden_nodes}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
