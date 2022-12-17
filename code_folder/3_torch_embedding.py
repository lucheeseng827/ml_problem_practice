import torch
import torch.nn as nn

# Create an embedding layer with 10 categories and a embedding size of 8
embedding = nn.Embedding(10, 8)

# Create a tensor with a batch of categorical data
data = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])

# Use the embedding layer to convert the categorical data into dense vectors
embedded = embedding(data)
print(embedded)
