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

# Define the data iterator (assuming you have `train_loader` and `test_loader`)
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# Train the model
for epoch in range(num_epochs):
    # Loop over the data iterator and process the inputs and labels
    for inputs, labels in train_loader:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()


# Function to evaluate the model
def evaluate(model, data_loader):
    model.eval()
    loss_total = 0
    correct_total = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)

            loss_total += loss.item()
            correct_total += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    loss_avg = loss_total / total_samples
    acc_avg = correct_total / total_samples
    return loss_avg, acc_avg


# Test the model
test_loss, test_acc = evaluate(model, test_loader)
print("Test Loss: {:.6f}, Test Acc: {:.6f}".format(test_loss, test_acc))
