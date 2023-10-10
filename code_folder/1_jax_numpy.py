import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from jax import random
from PIL import Image


# Define the model
class SimpleCNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(features=10)(x)  # Assuming 10 classes
        return x


train_state = SimpleCNN().init(random.PRNGKey(0), jnp.ones([1, 28, 28, 1]))

# Define the optimizer
optimizer = optax.adam(learning_rate=1e-3)


# Load dataset
def load_images_from_folder(folder):
    images = []
    labels = []  # Assuming each subfolder in the folder represents a class
    class_idx = 0
    for subdir in os.listdir(folder):
        for filename in os.listdir(os.path.join(folder, subdir)):
            img = Image.open(os.path.join(folder, subdir, filename))
            if img is not None:
                images.append(np.array(img))
                labels.append(class_idx)
        class_idx += 1
    return np.array(images), np.array(labels)


images, labels = load_images_from_folder("files")

# Normalize and preprocess the data
images = images.astype(jnp.float32) / 255.0
labels = jnp.array(labels, jnp.int32)


# Define the loss function
def cross_entropy_loss(logits, labels):
    one_hot = jax.nn.one_hot(labels, 10)
    return -jnp.mean(jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1))


# Training step
@jax.jit
def train_step(state, images, labels):
    def loss_fn(params):
        logits = SimpleCNN().apply({"params": params}, images)
        loss = cross_entropy_loss(logits, labels)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    return state.apply_gradients(grads=grads), loss


# Initialize model and optimizer
key = random.PRNGKey(0)
_, initial_params = SimpleCNN().init_by_shape(key, [((1, 32, 32, 3), jnp.float32)])
tx = optax.adam(0.001)
state = train_state.TrainState.create(
    apply_fn=SimpleCNN().apply, params=initial_params, tx=tx
)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    state, loss = train_step(state, images, labels)
    print(f"Epoch {epoch+1}, Loss: {loss}")

print("Training complete!")
