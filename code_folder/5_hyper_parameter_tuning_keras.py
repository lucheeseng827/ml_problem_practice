import tensorflow as tf
from tensorflow import keras


# Define the model
def build_model(input_size, hidden_size, num_classes, learning_rate):
    model = keras.Sequential()
    model.add(
        keras.layers.Dense(hidden_size, input_shape=(input_size,), activation="relu")
    )
    model.add(keras.layers.Dense(num_classes, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
        metrics=["accuracy"],
    )
    return model


# Define the hyperparameters
input_size = 28 * 28
hidden_size = 100
num_classes = 10
learning_rate = 0.001
num_epochs = 5
batch_size = 32

# Create the model
model = build_model(input_size, hidden_size, num_classes, learning_rate)

# Train the model
model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=num_epochs,
    verbose=1,
    validation_data=(x_val, y_val),
)

# Test the model
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
