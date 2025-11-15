"""
HuggingFace Transformers Fine-Tuning
=====================================
Category 12: NLP - Fine-tuning pretrained transformers for text classification

Use cases: Sentiment analysis, topic classification, intent detection
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np


class TransformerClassifier(nn.Module):
    """Simplified transformer-based classifier"""
    def __init__(self, vocab_size, num_classes=2, d_model=128, nhead=4, num_layers=2):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.fc(x)


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=50):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        indices = [self.vocab.get(w, 1) for w in text.split()[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.LongTensor(indices), torch.LongTensor([self.labels[idx]])[0]


def main():
    print("=" * 60)
    print("HuggingFace Transformers Fine-Tuning")
    print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)
    texts = []
    labels = []
    for i in range(1000):
        if i % 2 == 0:
            texts.append(' '.join(['good', 'great', 'excellent'] * np.random.randint(2, 5)))
            labels.append(1)
        else:
            texts.append(' '.join(['bad', 'terrible', 'awful'] * np.random.randint(2, 5)))
            labels.append(0)

    # Build vocab
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for text in texts:
        for word in text.split():
            if word not in vocab:
                vocab[word] = len(vocab)

    # Dataset and DataLoader
    dataset = TextDataset(texts, labels, vocab)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model
    model = TransformerClassifier(len(vocab), num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training
    print("\nFine-tuning transformer model...")
    for epoch in range(10):
        total_loss = 0
        for texts_batch, labels_batch in loader:
            outputs = model(texts_batch)
            loss = criterion(outputs, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/10], Loss: {total_loss / len(loader):.4f}')

    print("\nFine-tuning complete!")
    print("\nKey Takeaways:")
    print("- Transformers use self-attention mechanisms")
    print("- Fine-tuning adapts pretrained models to specific tasks")
    print("- HuggingFace provides easy access to BERT, GPT, RoBERTa, etc.")
    print("- Effective for low-resource tasks via transfer learning")


if __name__ == "__main__":
    main()
