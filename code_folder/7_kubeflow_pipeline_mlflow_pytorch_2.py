import kfp
import mlflow
import mlflow.pytorch
from kfp import dsl
from kubernetes.client import V1Toleration


def train_with_pytorch(epochs: int, learning_rate: float, momentum: float, dropout: float, hidden_size: int, batch_size: int, mlflow_experiment_name: str):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    from model import Net

    # Define the device to train the model on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define the transforms to apply to the data
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])

    # Load the training and test data
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    # Define the neural network
    net = Net(hidden_size, dropout)
    net.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    # Start the MLflow run
    with mlflow.start_run(experiment_id=mlflow_experiment_name, run_name="pytorch-training") as run:
        # Log the parameters
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("momentum", momentum)
        mlflow.log_param("dropout", dropout)
        mlflow.log_param("hidden_size", hidden_size)
        mlflow.log_param("batch_size", batch_size)

        # Log the model architecture
        mlflow.pytorch.log_model(net, "model")

        # Train the network
        for epoch in range(epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # send the inputs and labels to the GPU if available
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    mlflow.log_metric("training_loss", running_loss / 2000)
                    running_loss = 0.0

        print('Finished Training')

        # Test the network
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels =
