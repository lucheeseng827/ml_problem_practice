"""
Implicit Feedback ALS (Alternating Least Squares)
==================================================
Category 13: Recommendations from implicit feedback (clicks, views)

Use cases: YouTube views, Spotify plays, e-commerce browsing
"""

import numpy as np


class ImplicitALS:
    """ALS for implicit feedback data"""

    def __init__(self, n_factors=20, n_iterations=15, regularization=0.01):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.reg = regularization

    def fit(self, interaction_matrix, alpha=40):
        n_users, n_items = interaction_matrix.shape

        # Confidence matrix
        confidence = 1 + alpha * interaction_matrix

        # Initialize factors
        self.user_factors = np.random.normal(size=(n_users, self.n_factors))
        self.item_factors = np.random.normal(size=(n_items, self.n_factors))

        for iteration in range(self.n_iterations):
            # Update user factors
            for u in range(n_users):
                Cu = np.diag(confidence[u])
                A = self.item_factors.T @ Cu @ self.item_factors + self.reg * np.eye(self.n_factors)
                b = self.item_factors.T @ Cu @ (interaction_matrix[u] > 0)
                self.user_factors[u] = np.linalg.solve(A, b)

            # Update item factors
            for i in range(n_items):
                Ci = np.diag(confidence[:, i])
                A = self.user_factors.T @ Ci @ self.user_factors + self.reg * np.eye(self.n_factors)
                b = self.user_factors.T @ Ci @ (interaction_matrix[:, i] > 0)
                self.item_factors[i] = np.linalg.solve(A, b)

            if (iteration + 1) % 5 == 0:
                print(f'Iteration {iteration + 1}/{self.n_iterations}')

    def recommend(self, user_id, n=5):
        scores = self.user_factors[user_id] @ self.item_factors.T
        return np.argsort(scores)[::-1][:n]


def main():
    print("=" * 60)
    print("Implicit Feedback ALS Recommendations")
    print("=" * 60)

    # Generate implicit feedback (binary interactions)
    np.random.seed(42)
    n_users, n_items = 100, 50
    interactions = (np.random.rand(n_users, n_items) < 0.2).astype(float)

    print(f"\nInteraction matrix: {interactions.shape}")
    print(f"Interactions: {interactions.sum():.0f}")

    # Train ALS
    als = ImplicitALS(n_factors=20, n_iterations=15)
    print("\nTraining implicit ALS...")
    als.fit(interactions)

    # Recommendations
    user_id = 0
    recs = als.recommend(user_id, n=5)
    print(f"\nTop 5 recommendations for user {user_id}: {recs}")

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("- Implicit feedback: clicks, views (no explicit ratings)")
    print("- Confidence weighted by interaction frequency")
    print("- ALS alternates between user/item factor updates")
    print("- Scalable to massive datasets")
    print("- Used by Spotify, YouTube")


if __name__ == "__main__":
    main()
