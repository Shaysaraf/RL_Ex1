import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = 200
batch_size = 100
learning_rate = 1e-3

# MNIST Dataset
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

# Data Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Function to train and evaluate a model with loss tracking
def train_and_evaluate_with_plot(model, optimizer, criterion, num_epochs, train_loader, test_loader):
    # Track losses
    epoch_losses = []

    # Training Loop
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.view(-1, 28*28))
            labels = Variable(labels)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Record loss for the epoch
        epoch_losses.append(total_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

    # Evaluation Loop
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, 28*28)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model: {accuracy:.2f} %')

    return accuracy, epoch_losses

# Train and plot for each model
print("Training Simple Logistic Regression with SGD...")
model1 = nn.Linear(input_size, num_classes)
criterion1 = nn.CrossEntropyLoss()
optimizer1 = torch.optim.SGD(model1.parameters(), lr=learning_rate)
accuracy1, losses1 = train_and_evaluate_with_plot(model1, optimizer1, criterion1, num_epochs, train_loader, test_loader)

print("\nTraining Simple Logistic Regression with Adam...")
model2 = nn.Linear(input_size, num_classes)
criterion2 = nn.CrossEntropyLoss()
optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate)
accuracy2, losses2 = train_and_evaluate_with_plot(model2, optimizer2, criterion2, num_epochs, train_loader, test_loader)

print("\nTraining Deep Neural Network with Adam...")
hidden_size = 500
model3 = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, num_classes)
)
criterion3 = nn.CrossEntropyLoss()
optimizer3 = torch.optim.Adam(model3.parameters(), lr=learning_rate)
accuracy3, losses3 = train_and_evaluate_with_plot(model3, optimizer3, criterion3, num_epochs, train_loader, test_loader)

# Plot the training losses
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), losses1, label="Logistic Regression (SGD)")
plt.plot(range(1, num_epochs + 1), losses2, label="Logistic Regression (Adam)")
plt.plot(range(1, num_epochs + 1), losses3, label="Deep Neural Network (Adam)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Epochs")
plt.legend()
plt.grid()
plt.show()