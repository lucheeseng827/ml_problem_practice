import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Create a simple model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(32,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
# Here we just do a dummy training as an example
x_train = tf.random.normal(shape=(1000, 32))
y_train = tf.random.normal(shape=(1000, 10))
model.fit(x_train, y_train, epochs=5)

# Save the model in TensorFlow SavedModel format
model.save('./saved_model')
