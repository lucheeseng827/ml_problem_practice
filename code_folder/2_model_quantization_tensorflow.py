import tensorflow as tf
from tensorflow import keras

# Load the model
model = keras.models.load_model("path_to_your_model.h5")

# Convert the model to the TensorFlow Lite format with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Set the optimization flag
converter.optimizations = [tf.lite.Optimize.DEFAULT]


# Define a representative dataset if you have one. This step is optional but can improve quantization.
def representative_dataset_gen():
    for _ in range(num_calibration_steps):
        # Get sample input data as a numpy array in a method of your choosing.
        yield [input_data]


# Uncomment if you have a representative dataset
# converter.representative_dataset = representative_dataset_gen

# Convert the model
tflite_quant_model = converter.convert()

# Save the converted model
with open("quantized_model.tflite", "wb") as f:
    f.write(tflite_quant_model)

print("Quantized model saved!")
