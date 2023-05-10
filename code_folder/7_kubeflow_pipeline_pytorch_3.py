import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from kfp import dsl
from kfp.components import func_to_container_op
from kfp.v2.dsl import (ClassificationMetrics, Dataset, Input, Metrics, Model,
                        Output, component)


@func_to_container_op
def train_model_pytorch(
    epochs: int,
    lr: float,
    batch_size: int,
    model_path: Output[Model],
    metrics: Output[Metrics],
):
    # Define the device to train the model on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define the PyTorch transforms to use for data augmentation and normalization
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
        ]
    )

    # Download and load the CIFAR-10 training dataset
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    # Download and load the CIFAR-10 testing dataset
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # Define the PyTorch network architecture
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(nn.functional.relu(self.conv1(x)))
            x = self.pool(nn.functional.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = nn.functional.relu(self.fc1(x))
            x = nn.functional.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # Instantiate the PyTorch network
    net = Net()

    # Send the network to the GPU if available
    net.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    # Start the MLflow run
    with mlflow.start_run() as run:
        # Log the hyperparameters
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("lr", lr)
        mlflow.log_param("batch_size", batch_size)

        # Train the network
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # send the inputs and labels to the GPU if available
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward
                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print(
                        "[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000)
                    )
                    running_loss = 0.0
                    # Log the loss for the current epoch
                    mlflow.log_metric("loss", running_loss / len(trainloader))

                    # Test the network
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for data in testloader:
                            images, labels = data
                            images, labels = images.to(device), labels.to(device)
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()

                    # Log the accuracy for the current epoch
                    accuracy = 100 * correct / total
                    mlflow.log_metric("accuracy", accuracy)

                    # Save the trained model
                    torch.save(net.state_dict(), model_path)

                    # Create a Metrics object with the final metrics
                    metrics_data = {"accuracy": accuracy}
                    metrics = Metrics.from_dict(metrics_data)

                    # Output the Metrics object
                    metrics.save(metrics.get_path(metrics))
