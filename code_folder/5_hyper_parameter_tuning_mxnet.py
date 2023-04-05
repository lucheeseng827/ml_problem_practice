import mxnet as mx
import numpy as np


# Define the model
class Net(mx.gluon.HybridBlock):
    def __init__(self, input_size, hidden_size, num_classes, **kwargs):
        super(Net, self).__init__(**kwargs)
        with self.name_scope():
            self.fc1 = mx.gluon.nn.Dense(hidden_size)
            self.fc2 = mx.gluon.nn.Dense(num_classes)

    def hybrid_forward(self, F, x):
        x = self.fc1(x)
        x = F.Activation(x, act_type="relu")
        x = self.fc2(x)
        return x


# Define the hyperparameters
input_size = 28 * 28
hidden_size = 100
num_classes = 10
learning_rate = 0.001
num_epochs = 5
batch_size = 32

# Create the model
model = Net(input_size, hidden_size, num_classes)

# Initialize the parameters
model.collect_params().initialize(mx.init.Xavier())

# Define the loss function and the optimizer
loss_fn = mx.gluon.loss.SoftmaxCrossEntropyLoss()
trainer = mx.gluon.Trainer(
    model.collect_params(), "sgd", {"learning_rate": learning_rate}
)

# Loop over the data iterator and process the inputs and labels
for epoch in range(num_epochs):
    for inputs, labels in data_iterator:
        # Convert the inputs and labels to MXNet arrays
        inputs = mx.nd.array(inputs)
        labels = mx.nd.array(labels)

        # Forward pass
        with mx.autograd.record():
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

        # Backward pass
        loss.backward()

        # Update the weights
        trainer.step(batch_size)

# Test the model
test_loss, test_acc = evaluate(model, test_iterator)
print("Test Loss: {:.6f}, Test Acc: {:.6f}".format(test_loss, test_acc))
