import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. Data Preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True,
                               transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False,
                              transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# 2. Model Construction with One Hidden Layer (Baseline)
class MLP_OneHidden(nn.Module):
    def __init__(self):
        super(MLP_OneHidden, self).__init__()
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.output = nn.Linear(128, 10)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)
        return x

# 3. Model Construction with Two Hidden Layers
class MLP_TwoHidden(nn.Module):
    def __init__(self):
        super(MLP_TwoHidden, self).__init__()
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(28*28, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.output = nn.Linear(64, 10)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)
        return x

# 4. Training and Evaluation Functions
def train_and_evaluate(model, epochs=10):
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
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
        print(f'Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.4f}, Accuracy: {train_accuracy:.2f}%')

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

# 5. Run and Compare Both Models
epochs = 10
# Model with one hidden layer
model_one_hidden = MLP_OneHidden()
train_losses_one_hidden, test_loss_one_hidden, test_accuracy_one_hidden = train_and_evaluate(model_one_hidden, epochs)

# Model with two hidden layers
model_two_hidden = MLP_TwoHidden()
train_losses_two_hidden, test_loss_two_hidden, test_accuracy_two_hidden = train_and_evaluate(model_two_hidden, epochs)

# 6. Plot Results
plt.figure(figsize=(10, 5))

# Plot training losses
plt.subplot(1, 2, 1)
plt.plot(train_losses_one_hidden, label='One Hidden Layer')
plt.plot(train_losses_two_hidden, label='Two Hidden Layers')
plt.title('Training Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot test accuracy
plt.subplot(1, 2, 2)
plt.bar(['One Hidden Layer', 'Two Hidden Layers'], [test_accuracy_one_hidden, test_accuracy_two_hidden], color=['blue', 'green'])
plt.title('Test Accuracy Comparison')
plt.ylabel('Accuracy (%)')

plt.tight_layout()
plt.show()

print(f"One Hidden Layer Test Loss: {test_loss_one_hidden:.4f}, Test Accuracy: {test_accuracy_one_hidden:.2f}%")
print(f"Two Hidden Layers Test Loss: {test_loss_two_hidden:.4f}, Test Accuracy: {test_accuracy_two_hidden:.2f}%")
