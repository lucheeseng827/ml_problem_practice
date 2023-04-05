import dagshub
import numpy as np
import tensorflow as tf

# Initialize DAGsHub experiment logger
dagshub.init()

# Define hyperparameters
learning_rate = 0.01
batch_size = 32
num_epochs = 10

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# Define model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Train model and log metrics to DAGsHub
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=num_epochs,
    validation_data=(x_test, y_test),
)
with dagshub.dagshub_logger() as logger:
    logger.log_hyperparams(
        {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
        }
    )
    logger.log_metrics(
        {
            "train_loss": history.history["loss"][-1],
            "train_acc": history.history["accuracy"][-1],
            "val_loss": history.history["val_loss"][-1],
            "val_acc": history.history["val_accuracy"][-1],
        }
    )
