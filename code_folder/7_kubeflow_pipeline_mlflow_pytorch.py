import kfp.dsl as dsl


# Define the PyTorch training component
@dsl.python_component(
    name="train_pytorch",
    description="Train a PyTorch model and log metrics with MLflow",
    base_image="pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime",
)
def train_pytorch(
    epochs: int,
    learning_rate: float,
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
    model_path: str,
):
    # Import necessary libraries
    import mlflow.pytorch
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms

    # Define the transform for the input data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Download the MNIST dataset and load it into dataloaders
    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=2
    )

    # Define the neural network architecture
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 4 * 4, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(-1, 16 * 4 * 4)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    # Set the device to the GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # Start the MLflow run
    with mlflow.start_run(
        experiment_id=mlflow_experiment_name, run_name="pytorch-training"
    ) as run:
        # Train the network
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # send the inputs and labels to the GPU if available
                inputs, labels = inputs.to(device), labels.to(device)

                # zero
                # forward + backward + optimize
                optimizer.zero_grad()
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
                    # Log the loss metric to MLflow
                    mlflow.log_metric("loss", running_loss / 2000)
                    running_loss = 0.0

        print("Finished Training")

        # Save the PyTorch model
        model_path = "models/pytorch_cifar_net.pth"
        torch.save(net.state_dict(), model_path)

    # Log the PyTorch model as an MLflow artifact
    mlflow.log_artifact(model_path, artifact_path="pytorch-model")


# Define a function to test the trained PyTorch model
def test_model():
    # Load the trained PyTorch model
    model_path = "models/pytorch_cifar_net.pth"
    net = Net()
    net.load_state_dict(torch.load(model_path))

    # Load the CIFAR-10 test dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )

    # Send the model and test data to the GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Test the model on the test dataset
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

    # Calculate the accuracy and log it as an MLflow metric
    accuracy = 100 * correct / total
    mlflow.log_metric("accuracy", accuracy)
    print("Accuracy of the network on the 10000 test images: %d %%" % accuracy)


# Define a Kubeflow pipeline that trains and tests the PyTorch model
@kfp.dsl.pipeline(name="pytorch-cifar10-mlflow-pipeline")
def pytorch_cifar10_mlflow_pipeline():
    train = kfp.dsl.ContainerOp(
        name="train",
        image="pytorch:latest",
        command=["sh", "-c"],
        arguments=["python train.py"],
        file_outputs={"model_path": "/mlflow-artifacts/pytorch-model"},
    )

    test = kfp.dsl.ContainerOp(
        name="test",
        image="pytorch:latest",
        command=["sh", "-c"],
        arguments=["python test.py"],
    ).after(train)
