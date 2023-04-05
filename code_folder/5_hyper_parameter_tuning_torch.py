import torch
import torch.nn as nn
import torch.optim as optim


# Define the model
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Define the hyperparameters
input_size = 28 * 28
hidden_size = 100
num_classes = 10
learning_rate = 0.001
num_epochs = 5

# Create the model, loss function, and optimizer
model = Net(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    # Loop over the data iterator and process the inputs and labels
    for inputs, labels in data_iterator:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

# Test the model
test_loss, test_acc = evaluate(model, test_iterator)
print("Test Loss: {:.6f}, Test Acc: {:.6f}".format(test_loss, test_acc))
