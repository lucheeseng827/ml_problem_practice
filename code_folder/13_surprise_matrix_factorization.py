"""
Matrix Factorization with Surprise Library
===========================================
Category 13: Recommender Systems - SVD, NMF for recommendation

Use cases: Netflix-style recommendations, rating prediction
"""

import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error


class MatrixFactorization:
    """Simple matrix factorization using gradient descent"""

    def __init__(self, n_factors=10, learning_rate=0.01, n_epochs=50, regularization=0.02):
        self.n_factors = n_factors
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.reg = regularization

    def fit(self, ratings_matrix):
        n_users, n_items = ratings_matrix.shape

        # Initialize latent factors
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))

        # Get non-zero entries
        user_indices, item_indices = np.where(ratings_matrix > 0)

        # Training
        for epoch in range(self.n_epochs):
            for idx in range(len(user_indices)):
                u = user_indices[idx]
                i = item_indices[idx]
                rating = ratings_matrix[u, i]

                # Prediction
                pred = np.dot(self.user_factors[u], self.item_factors[i])

                # Error
                error = rating - pred

                # Update factors
                self.user_factors[u] += self.lr * (error * self.item_factors[i] - self.reg * self.user_factors[u])
                self.item_factors[i] += self.lr * (error * self.user_factors[u] - self.reg * self.item_factors[i])

            if (epoch + 1) % 10 == 0:
                pred_matrix = np.dot(self.user_factors, self.item_factors.T)
                mse = mean_squared_error(ratings_matrix[ratings_matrix > 0], pred_matrix[ratings_matrix > 0])
                print(f'Epoch {epoch + 1}, MSE: {mse:.4f}')

    def predict(self, user_id, item_id):
        return np.dot(self.user_factors[user_id], self.item_factors[item_id])

    def recommend(self, user_id, top_n=5):
        predictions = np.dot(self.user_factors[user_id], self.item_factors.T)
        return np.argsort(predictions)[::-1][:top_n]


def main():
    print("=" * 60)
    print("Matrix Factorization for Recommender Systems")
    print("=" * 60)

    # Generate ratings
    np.random.seed(42)
    n_users, n_items = 100, 50
    ratings = np.random.randint(1, 6, size=(n_users, n_items)).astype(float)
    mask = np.random.rand(n_users, n_items) < 0.7
    ratings[mask] = 0

    print(f"\nRatings matrix: {ratings.shape}")

    # Matrix Factorization
    mf = MatrixFactorization(n_factors=10, n_epochs=50)
    print("\nTraining matrix factorization...")
    mf.fit(ratings)

    # Recommendations
    user_id = 0
    recommendations = mf.recommend(user_id, top_n=5)
    print(f"\nTop 5 recommendations for user {user_id}: {recommendations}")

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("- MF decomposes ratings matrix into user/item latent factors")
    print("- SVD: optimal low-rank approximation")
    print("- SGD optimization for large sparse matrices")
    print("- Regularization prevents overfitting")
    print("- Scalable to millions of users/items")


if __name__ == "__main__":
    main()
