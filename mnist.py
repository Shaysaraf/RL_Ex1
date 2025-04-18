import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = 20
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

# Data Loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Function to train and evaluate a model
def train_and_evaluate(model, optimizer, criterion, num_epochs, train_loader, test_loader):
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
    return accuracy

# 1. Simple Logistic Regression with SGD
print("Training Simple Logistic Regression with SGD...")
model1 = nn.Linear(input_size, num_classes)
criterion1 = nn.CrossEntropyLoss()
optimizer1 = torch.optim.SGD(model1.parameters(), lr=learning_rate)
accuracy1 = train_and_evaluate(model1, optimizer1, criterion1, num_epochs, train_loader, test_loader)

# 2. Simple Logistic Regression with Adam
print("\nTraining Simple Logistic Regression with Adam...")
model2 = nn.Linear(input_size, num_classes)
criterion2 = nn.CrossEntropyLoss()
optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate)
accuracy2 = train_and_evaluate(model2, optimizer2, criterion2, num_epochs, train_loader, test_loader)

# 3. Deep Neural Network with Adam
print("\nTraining Deep Neural Network with Adam...")
hidden_size = 500
model3 = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, num_classes)
)
criterion3 = nn.CrossEntropyLoss()
optimizer3 = torch.optim.Adam(model3.parameters(), lr=learning_rate)
accuracy3 = train_and_evaluate(model3, optimizer3, criterion3, num_epochs, train_loader, test_loader)

# Print final results
print("\nFinal Results:")
print(f"Simple Logistic Regression with SGD Accuracy: {accuracy1:.2f} %")
print(f"Simple Logistic Regression with Adam Accuracy: {accuracy2:.2f} %")
print(f"Deep Neural Network with Adam Accuracy: {accuracy3:.2f} %")
