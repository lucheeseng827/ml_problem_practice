"""
Text Classification with TF-IDF and Scikit-Learn
=================================================
Category 12: Natural Language Processing

This example demonstrates:
- Text preprocessing and cleaning
- TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
- Traditional ML classifiers for text
- N-grams and feature engineering
- Pipeline construction for text processing

Use cases:
- Spam detection
- Sentiment analysis
- Document categorization
- Content moderation
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import re


def generate_synthetic_text_data(n_samples=2000):
    """Generate synthetic text classification dataset"""
    np.random.seed(42)

    categories = ['sports', 'technology', 'politics', 'entertainment', 'business']

    # Vocabulary for each category
    vocab = {
        'sports': ['game', 'team', 'player', 'score', 'win', 'championship', 'coach', 'training', 'match', 'league'],
        'technology': ['software', 'computer', 'AI', 'algorithm', 'code', 'developer', 'app', 'digital', 'data', 'cloud'],
        'politics': ['government', 'election', 'policy', 'vote', 'president', 'congress', 'law', 'campaign', 'senator', 'bill'],
        'entertainment': ['movie', 'actor', 'music', 'film', 'concert', 'show', 'celebrity', 'performance', 'album', 'director'],
        'business': ['market', 'stock', 'company', 'profit', 'sales', 'investor', 'revenue', 'CEO', 'corporate', 'finance']
    }

    texts = []
    labels = []

    for _ in range(n_samples):
        category = np.random.choice(categories)
        category_words = vocab[category]

        # Generate text with 10-20 words
        n_words = np.random.randint(10, 21)
        words = []

        for _ in range(n_words):
            # 70% from category vocab, 30% from other categories
            if np.random.rand() < 0.7:
                word = np.random.choice(category_words)
            else:
                other_category = np.random.choice([c for c in categories if c != category])
                word = np.random.choice(vocab[other_category])

            words.append(word)

        text = ' '.join(words)
        texts.append(text)
        labels.append(category)

    return texts, labels


def preprocess_text(text):
    """Clean and preprocess text"""
    # Lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def tfidf_classification_example():
    """TF-IDF with multiple classifiers"""
    print("=" * 60)
    print("Text Classification with TF-IDF")
    print("=" * 60)

    # Generate data
    print("\nGenerating synthetic text dataset...")
    texts, labels = generate_synthetic_text_data(n_samples=2000)

    # Preprocess
    texts = [preprocess_text(text) for text in texts]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Categories: {set(labels)}")

    # TF-IDF Vectorizer
    tfidf = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=2,  # Minimum document frequency
        max_df=0.8,  # Maximum document frequency
        stop_words='english'
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    print(f"\nTF-IDF matrix shape: {X_train_tfidf.shape}")
    print(f"Vocabulary size: {len(tfidf.vocabulary_)}")

    # Train multiple classifiers
    classifiers = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Linear SVM': LinearSVC(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}

    print("\nTraining classifiers...")
    for name, clf in classifiers.items():
        # Train
        clf.fit(X_train_tfidf, y_train)

        # Predict
        y_pred = clf.predict(X_test_tfidf)

        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy

        print(f"\n{name}:")
        print(f"  Accuracy: {accuracy:.4f}")

        # Detailed report for best model
        if name == 'Logistic Regression':
            print("\nClassification Report (Logistic Regression):")
            print(classification_report(y_test, y_pred))

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=list(set(labels)))

            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=sorted(set(labels)),
                       yticklabels=sorted(set(labels)))
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix - Logistic Regression')
            plt.tight_layout()
            plt.savefig('/tmp/tfidf_confusion_matrix.png')
            print("\nConfusion matrix saved to /tmp/tfidf_confusion_matrix.png")

    # Visualize results
    plt.figure(figsize=(10, 6))
    bars = plt.bar(results.keys(), results.values())
    plt.xlabel('Classifier')
    plt.ylabel('Accuracy')
    plt.title('Text Classification Performance Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.grid(True, axis='y', alpha=0.3)

    # Color bars
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # Add value labels
    for i, (name, acc) in enumerate(results.items()):
        plt.text(i, acc + 0.02, f'{acc:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('/tmp/tfidf_comparison.png')
    print("\nComparison plot saved to /tmp/tfidf_comparison.png")

    return classifiers['Logistic Regression'], tfidf


def ngram_analysis():
    """Analyze different n-gram ranges"""
    print("\n" + "=" * 60)
    print("N-gram Analysis")
    print("=" * 60)

    texts, labels = generate_synthetic_text_data(n_samples=1000)
    texts = [preprocess_text(text) for text in texts]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    ngram_configs = [
        (1, 1),  # Unigrams only
        (1, 2),  # Unigrams + Bigrams
        (1, 3),  # Unigrams + Bigrams + Trigrams
        (2, 2),  # Bigrams only
        (2, 3),  # Bigrams + Trigrams
    ]

    results = {}

    print("\nTesting different n-gram ranges...")
    for ngram_range in ngram_configs:
        # Vectorize
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=500,
            stop_words='english'
        )

        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Train classifier
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train_vec, y_train)

        # Evaluate
        accuracy = clf.score(X_test_vec, y_test)
        results[str(ngram_range)] = accuracy

        print(f"N-gram range {ngram_range}: Accuracy = {accuracy:.4f}")

    # Visualize
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values(), color='steelblue')
    plt.xlabel('N-gram Range')
    plt.ylabel('Accuracy')
    plt.title('Impact of N-gram Range on Classification Accuracy')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('/tmp/ngram_analysis.png')
    print("\nN-gram analysis saved to /tmp/ngram_analysis.png")


def feature_importance_analysis(model, vectorizer, top_n=10):
    """Analyze feature importance for each class"""
    print("\n" + "=" * 60)
    print("Feature Importance Analysis")
    print("=" * 60)

    feature_names = vectorizer.get_feature_names_out()
    categories = model.classes_

    print(f"\nTop {top_n} features for each category:")

    for i, category in enumerate(categories):
        # Get coefficients for this class
        coef = model.coef_[i]

        # Get top features
        top_indices = np.argsort(coef)[-top_n:][::-1]
        top_features = [feature_names[idx] for idx in top_indices]
        top_scores = [coef[idx] for idx in top_indices]

        print(f"\n{category.upper()}:")
        for feat, score in zip(top_features, top_scores):
            print(f"  {feat:20s}: {score:.4f}")


def pipeline_example():
    """Complete text classification pipeline"""
    print("\n" + "=" * 60)
    print("Complete Text Classification Pipeline")
    print("=" * 60)

    texts, labels = generate_synthetic_text_data(n_samples=1500)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2
        )),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])

    print("\nPipeline steps:")
    for name, step in pipeline.steps:
        print(f"  {name}: {step.__class__.__name__}")

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluate
    accuracy = pipeline.score(X_test, y_test)
    print(f"\nPipeline accuracy: {accuracy:.4f}")

    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Example predictions
    sample_texts = [
        "the team won the championship game",
        "new software update released today",
        "government announces new policy"
    ]

    predictions = pipeline.predict(sample_texts)

    print("\nExample Predictions:")
    for text, pred in zip(sample_texts, predictions):
        print(f"  '{text}' -> {pred}")

    return pipeline


def main():
    """Main execution function"""
    print("Text Classification with TF-IDF and Scikit-Learn\n")

    # Example 1: Basic classification with multiple models
    model, vectorizer = tfidf_classification_example()

    # Example 2: N-gram analysis
    ngram_analysis()

    # Example 3: Feature importance
    feature_importance_analysis(model, vectorizer, top_n=10)

    # Example 4: Complete pipeline
    pipeline = pipeline_example()

    print("\n" + "=" * 60)
    print("Text Classification Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("- TF-IDF captures word importance across documents")
    print("- N-grams capture phrase patterns")
    print("- Logistic Regression performs well for text")
    print("- Naive Bayes is fast and effective for text")
    print("- Pipelines streamline preprocessing and training")
    print("- Feature analysis reveals important keywords")


if __name__ == "__main__":
    main()
