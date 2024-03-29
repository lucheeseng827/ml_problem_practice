import ray
import torch
import torch.nn as nn
from ray.air.config import ScalingConfig
from ray.train.torch import TorchTrainer

# If using GPUs, set this to True.
use_gpu = False


class NeuralNetwork(nn.Module):
    """
    Neural network model for regression.

    Args:
        input_size (int): Size of the input features.
        layer_size (int): Size of the hidden layer.
        output_size (int): Size of the output.
    """

    def __init__(self, input_size, layer_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, layer_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(layer_size, output_size)

    def forward(self, input):
        """
        Forward pass of the neural network.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.layer2(self.relu(self.layer1(input)))


def train_loop_per_worker():
    """
    Training loop for each worker.

    This function is executed by each worker during training.

    It performs the forward and backward passes, updates the model parameters,
    and reports the training progress.

    Raises:
        ValueError: If the dataset shard is not found.
    """

    dataset_shard = session.get_dataset_shard("train")
    if dataset_shard is None:
        raise ValueError("Dataset shard 'train' not found.")

    model = NeuralNetwork(input_size=1, layer_size=15, output_size=1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    model = train.torch.prepare_model(model)

    for epoch in range(num_epochs):
        for batches in dataset_shard.iter_torch_batches(
            batch_size=32, dtypes=torch.float
        ):
            inputs, labels = torch.unsqueeze(batches["x"], 1), batches["y"]
            output = model(inputs)
            loss = loss_fn(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"epoch: {epoch}, loss: {loss.item()}")

        session.report(
            {},
            checkpoint=Checkpoint.from_dict(
                dict(epoch=epoch, model=model.state_dict())
            ),
        )


train_dataset = ray.data.from_items([{"x": x, "y": 2 * x + 1} for x in range(200)])
scaling_config = ScalingConfig(num_workers=3, use_gpu=use_gpu)
trainer = TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,
    scaling_config=scaling_config,
    datasets={"train": train_dataset},
)
result = trainer.fit()
