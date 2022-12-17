from tensorflow.keras.layers import Embedding

# Define the input layer for a sequence of words
input_layer = Input(shape=(None,))

# Add an embedding layer with a vocabulary of 10000 words and a embedding dimension of 100
embedding_layer = Embedding(10000, 100)(input_layer)

# Add other layers and compile the model
...
