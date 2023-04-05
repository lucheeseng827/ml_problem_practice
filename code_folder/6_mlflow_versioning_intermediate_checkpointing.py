import os
import tempfile
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from google.cloud import storage

# Set up Google Cloud Storage
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/your/credentials.json"
bucket_name = "your_bucket_name"
storage_client = storage.Client()
bucket = storage_client.get_bucket(bucket_name)

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build the model
model = Sequential(
    [
        Flatten(input_shape=(28, 28)),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(10, activation="softmax"),
    ]
)

# Compile the model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Set up MLflow tracking and automatic logging
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mnist_experiment")
mlflow.tensorflow.autolog()

# Define a checkpoint callback
checkpoint_dir = tempfile.mkdtemp()
checkpoint_path = os.path.join(checkpoint_dir, "model_checkpoint_{epoch:02d}.hdf5")

checkpoint_callback = ModelCheckpoint(
    checkpoint_path, save_freq="epoch", verbose=1, save_weights_only=True
)

# Train the model
model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_data=(x_test, y_test),
    callbacks=[checkpoint_callback],
)

# Upload model checkpoints to Google Cloud Storage
for file in os.listdir(checkpoint_dir):
    file_path = os.path.join(checkpoint_dir, file)
    blob = bucket.blob(f"model_checkpoints/{file}")
    blob.upload_from_filename(file_path)

# Cleanup temporary checkpoint directory
os.rmdir(checkpoint_dir)
