"""
Sentiment Analysis with LSTM
=============================
Category 12: Natural Language Processing

This example demonstrates:
- Text tokenization and embedding
- LSTM networks for sequence classification
- Bidirectional LSTM
- Word embeddings (Word2Vec, GloVe concepts)
- Sentiment classification

Use cases:
- Customer review analysis
- Social media monitoring
- Brand sentiment tracking
- Product feedback analysis
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


class SentimentDataset(Dataset):
    """Dataset for sentiment analysis"""

    def __init__(self, texts, labels, vocab, max_len=50):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Convert text to indices
        indices = [self.vocab.get(word, 0) for word in text.split()]

        # Pad or truncate
        if len(indices) < self.max_len:
            indices += [0] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]

        return torch.LongTensor(indices), torch.LongTensor([label])[0]


class LSTMSentimentClassifier(nn.Module):
    """LSTM model for sentiment classification"""

    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, num_layers=2, dropout=0.3):
        super(LSTMSentimentClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 2)  # Binary: positive/negative

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # Use last hidden state
        final_hidden = hidden[-1]
        dropped = self.dropout(final_hidden)
        output = self.fc(dropped)

        return output


class BiLSTMSentimentClassifier(nn.Module):
    """Bidirectional LSTM for sentiment analysis"""

    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, num_layers=2, dropout=0.3):
        super(BiLSTMSentimentClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bilstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 2)  # *2 for bidirectional

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.bilstm(embedded)

        # Concatenate final forward and backward hidden states
        hidden_fwd = hidden[-2]
        hidden_bwd = hidden[-1]
        final_hidden = torch.cat([hidden_fwd, hidden_bwd], dim=1)

        dropped = self.dropout(final_hidden)
        output = self.fc(dropped)

        return output


def generate_sentiment_data(n_samples=2000):
    """Generate synthetic sentiment data"""
    np.random.seed(42)

    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                     'love', 'best', 'perfect', 'awesome', 'brilliant', 'outstanding']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'poor',
                     'hate', 'disappointing', 'useless', 'pathetic', 'dreadful', 'disgusting']
    neutral_words = ['the', 'a', 'is', 'was', 'it', 'this', 'that', 'with', 'for', 'on']

    texts = []
    labels = []

    for _ in range(n_samples):
        sentiment = np.random.choice([0, 1])  # 0: negative, 1: positive

        if sentiment == 1:
            # Positive review
            n_positive = np.random.randint(3, 6)
            n_neutral = np.random.randint(3, 6)
            words = (np.random.choice(positive_words, n_positive).tolist() +
                    np.random.choice(neutral_words, n_neutral).tolist())
        else:
            # Negative review
            n_negative = np.random.randint(3, 6)
            n_neutral = np.random.randint(3, 6)
            words = (np.random.choice(negative_words, n_negative).tolist() +
                    np.random.choice(neutral_words, n_neutral).tolist())

        np.random.shuffle(words)
        text = ' '.join(words)

        texts.append(text)
        labels.append(sentiment)

    return texts, labels


def build_vocabulary(texts):
    """Build vocabulary from texts"""
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2

    for text in texts:
        for word in text.split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1

    return vocab


def train_sentiment_classifier():
    """Train LSTM sentiment classifier"""
    print("=" * 60)
    print("LSTM Sentiment Analysis Training")
    print("=" * 60)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on: {device}")

    # Generate data
    print("\nGenerating sentiment data...")
    texts, labels = generate_sentiment_data(n_samples=2000)

    # Build vocabulary
    vocab = build_vocabulary(texts)
    print(f"Vocabulary size: {len(vocab)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Create datasets
    train_dataset = SentimentDataset(X_train, y_train, vocab, max_len=20)
    test_dataset = SentimentDataset(X_test, y_test, vocab, max_len=20)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model
    model = LSTMSentimentClassifier(
        vocab_size=len(vocab),
        embedding_dim=100,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3
    ).to(device)

    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    num_epochs = 20
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    print("\nTraining LSTM model...")
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        train_preds, train_targets = [], []

        for texts_batch, labels_batch in train_loader:
            texts_batch = texts_batch.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(texts_batch)
            loss = criterion(outputs, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(labels_batch.cpu().numpy())

        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_targets, train_preds)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Test
        model.eval()
        test_loss = 0
        test_preds, test_targets = [], []

        with torch.no_grad():
            for texts_batch, labels_batch in test_loader:
                texts_batch = texts_batch.to(device)
                labels_batch = labels_batch.to(device)

                outputs = model(texts_batch)
                loss = criterion(outputs, labels_batch)

                test_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                test_preds.extend(preds.cpu().numpy())
                test_targets.extend(labels_batch.cpu().numpy())

        test_loss /= len(test_loader)
        test_acc = accuracy_score(test_targets, test_preds)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}]')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

    # Final evaluation
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_targets, test_preds, target_names=['Negative', 'Positive']))

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(test_losses, label='Test Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training History - Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(train_accs, label='Train Accuracy')
    axes[1].plot(test_accs, label='Test Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training History - Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('/tmp/lstm_sentiment_training.png')
    print("\nTraining plot saved to /tmp/lstm_sentiment_training.png")

    # Save model
    torch.save(model.state_dict(), '/tmp/lstm_sentiment_model.pth')
    print("Model saved to /tmp/lstm_sentiment_model.pth")

    return model, vocab


def train_bilstm_classifier():
    """Train Bidirectional LSTM"""
    print("\n" + "=" * 60)
    print("Bidirectional LSTM Training")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    texts, labels = generate_sentiment_data(n_samples=2000)
    vocab = build_vocabulary(texts)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    train_dataset = SentimentDataset(X_train, y_train, vocab, max_len=20)
    test_dataset = SentimentDataset(X_test, y_test, vocab, max_len=20)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # BiLSTM model
    model = BiLSTMSentimentClassifier(
        vocab_size=len(vocab),
        embedding_dim=100,
        hidden_dim=128,
        num_layers=2
    ).to(device)

    print(f"BiLSTM Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train for 15 epochs
    print("\nTraining BiLSTM...")
    for epoch in range(15):
        model.train()
        for texts_batch, labels_batch in train_loader:
            texts_batch = texts_batch.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(texts_batch)
            loss = criterion(outputs, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for texts_batch, labels_batch in test_loader:
                    texts_batch = texts_batch.to(device)
                    labels_batch = labels_batch.to(device)
                    outputs = model(texts_batch)
                    _, predicted = torch.max(outputs, 1)
                    total += labels_batch.size(0)
                    correct += (predicted == labels_batch).sum().item()

            accuracy = correct / total
            print(f'Epoch [{epoch + 1}/15], Test Accuracy: {accuracy:.4f}')

    print("\nBiLSTM training complete!")

    return model


def main():
    """Main execution function"""
    print("Sentiment Analysis with PyTorch LSTM\n")

    # Train LSTM
    lstm_model, vocab = train_sentiment_classifier()

    # Train BiLSTM
    bilstm_model = train_bilstm_classifier()

    print("\n" + "=" * 60)
    print("Sentiment Analysis Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("- LSTMs capture sequential dependencies in text")
    print("- Embeddings convert words to dense vectors")
    print("- BiLSTM processes sequences in both directions")
    print("- Dropout prevents overfitting")
    print("- Sentiment analysis is binary/multi-class classification")


if __name__ == "__main__":
    main()
