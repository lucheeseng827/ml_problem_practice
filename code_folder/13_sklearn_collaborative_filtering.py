"""
Collaborative Filtering with Scikit-Learn
==========================================
Category 13: Recommender Systems - User-based and item-based collaborative filtering

Use cases: Movie/product recommendations, content discovery, personalization
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error


def generate_ratings_matrix(n_users=100, n_items=50, sparsity=0.7):
    """Generate synthetic user-item ratings matrix"""
    np.random.seed(42)
    ratings = np.random.randint(1, 6, size=(n_users, n_items)).astype(float)
    # Make sparse
    mask = np.random.rand(n_users, n_items) < sparsity
    ratings[mask] = 0
    return ratings


def user_based_collaborative_filtering(ratings_matrix, user_id, top_n=5):
    """User-based collaborative filtering"""
    # Compute user similarity
    user_similarity = cosine_similarity(ratings_matrix)

    # Get similar users
    similar_users = np.argsort(user_similarity[user_id])[::-1][1:top_n + 1]

    # Get recommendations
    user_ratings = ratings_matrix[user_id]
    recommendations = np.zeros(ratings_matrix.shape[1])

    for similar_user in similar_users:
        similarity_score = user_similarity[user_id, similar_user]
        recommendations += ratings_matrix[similar_user] * similarity_score

    # Filter already rated items
    recommendations[user_ratings > 0] = 0

    return np.argsort(recommendations)[::-1][:top_n]


def item_based_collaborative_filtering(ratings_matrix, user_id, top_n=5):
    """Item-based collaborative filtering"""
    # Compute item similarity
    item_similarity = cosine_similarity(ratings_matrix.T)

    # Get user's rated items
    user_ratings = ratings_matrix[user_id]
    rated_items = np.where(user_ratings > 0)[0]

    # Predict ratings for unrated items
    predictions = np.zeros(ratings_matrix.shape[1])

    for item in range(ratings_matrix.shape[1]):
        if user_ratings[item] == 0:  # Unrated
            similar_items = rated_items
            similarities = item_similarity[item, similar_items]
            ratings = user_ratings[similar_items]

            if np.sum(np.abs(similarities)) > 0:
                predictions[item] = np.dot(similarities, ratings) / np.sum(np.abs(similarities))

    return np.argsort(predictions)[::-1][:top_n]


def main():
    print("=" * 60)
    print("Collaborative Filtering Recommender Systems")
    print("=" * 60)

    # Generate ratings matrix
    ratings = generate_ratings_matrix(n_users=100, n_items=50, sparsity=0.7)
    print(f"\nRatings matrix shape: {ratings.shape}")
    print(f"Sparsity: {(ratings == 0).sum() / ratings.size:.2%}")

    # User-based CF
    user_id = 0
    user_recs = user_based_collaborative_filtering(ratings, user_id, top_n=5)
    print(f"\nUser-based recommendations for user {user_id}: {user_recs}")

    # Item-based CF
    item_recs = item_based_collaborative_filtering(ratings, user_id, top_n=5)
    print(f"Item-based recommendations for user {user_id}: {item_recs}")

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("- User-based: Find similar users, recommend their items")
    print("- Item-based: Find similar items to what user liked")
    print("- Cosine similarity measures user/item similarity")
    print("- Handles cold-start problem poorly")
    print("- Scalability challenges with large datasets")


if __name__ == "__main__":
    main()
