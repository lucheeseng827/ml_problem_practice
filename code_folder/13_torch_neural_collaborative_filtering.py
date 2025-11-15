"""
Neural Collaborative Filtering with PyTorch
============================================
Category 13: Deep learning for recommendations

Use cases: Deep personalization, complex user-item interactions
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


class NCFDataset(Dataset):
    def __init__(self, ratings_matrix):
        self.user_ids, self.item_ids = np.where(ratings_matrix > 0)
        self.ratings = ratings_matrix[self.user_ids, self.item_ids]

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]


class NeuralCF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=50):
        super(NeuralCF, self).__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, user_ids, item_ids):
        user_embed = self.user_embedding(user_ids)
        item_embed = self.item_embedding(item_ids)
        x = torch.cat([user_embed, item_embed], dim=1)
        return self.fc_layers(x)


def main():
    print("=" * 60)
    print("Neural Collaborative Filtering")
    print("=" * 60)

    np.random.seed(42)
    n_users, n_items = 100, 50
    ratings = np.random.randint(1, 6, (n_users, n_items)).astype(float)
    mask = np.random.rand(n_users, n_items) < 0.7
    ratings[mask] = 0

    dataset = NCFDataset(ratings)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = NeuralCF(n_users, n_items, embedding_dim=50)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print("\nTraining neural CF...")
    for epoch in range(20):
        total_loss = 0
        for user_ids, item_ids, ratings_batch in loader:
            user_ids = user_ids.long()
            item_ids = item_ids.long()
            ratings_batch = ratings_batch.float()

            predictions = model(user_ids, item_ids).squeeze()
            loss = criterion(predictions, ratings_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/20], Loss: {total_loss / len(loader):.4f}')

    print("\nKey Takeaways:")
    print("- Neural CF learns non-linear user-item interactions")
    print("- Embeddings capture latent preferences")
    print("- Can incorporate side information (features)")
    print("- More expressive than linear matrix factorization")


if __name__ == "__main__":
    main()
