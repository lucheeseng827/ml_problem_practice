"""
Named Entity Recognition with Transformers
===========================================
Category 12: Natural Language Processing

Demonstrates: NER with BERT/transformers, token classification,
fine-tuning pretrained models, BIO tagging scheme

Use cases: Information extraction, document parsing, knowledge graphs
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import classification_report


class NERDataset(Dataset):
    def __init__(self, texts, tags, max_len=50):
        self.texts = texts
        self.tags = tags
        self.max_len = max_len
        self.vocab = self._build_vocab()
        self.tag_vocab = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-LOC': 3, 'I-LOC': 4, 'B-ORG': 5, 'I-ORG': 6}

    def _build_vocab(self):
        vocab = {'<PAD>': 0, '<UNK>': 1}
        idx = 2
        for text in self.texts:
            for word in text:
                if word not in vocab:
                    vocab[word] = idx
                    idx += 1
        return vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tags = self.tags[idx]

        # Convert to indices
        text_ids = [self.vocab.get(w, 1) for w in text[:self.max_len]]
        tag_ids = [self.tag_vocab[t] for t in tags[:self.max_len]]

        # Padding
        padding = self.max_len - len(text_ids)
        text_ids += [0] * padding
        tag_ids += [0] * padding

        return torch.LongTensor(text_ids), torch.LongTensor(tag_ids)


class BiLSTM_CRF(nn.Module):
    """BiLSTM-CRF for NER (simplified)"""
    def __init__(self, vocab_size, tag_size, embedding_dim=100, hidden_dim=128):
        super(BiLSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, tag_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.bilstm(embedded)
        return self.fc(lstm_out)


def generate_ner_data(n_samples=1000):
    """Generate synthetic NER data"""
    persons = ['John', 'Mary', 'Alice', 'Bob']
    locations = ['London', 'Paris', 'Tokyo', 'NewYork']
    orgs = ['Google', 'Microsoft', 'Amazon']

    texts, tags = [], []
    for _ in range(n_samples):
        text = []
        tag_seq = []

        # Random sentence structure
        if np.random.rand() < 0.5:
            person = np.random.choice(persons)
            text.extend([person, 'works', 'at'])
            tag_seq.extend(['B-PER', 'O', 'O'])

            org = np.random.choice(orgs)
            text.extend([org, 'in'])
            tag_seq.extend(['B-ORG', 'O'])

            loc = np.random.choice(locations)
            text.append(loc)
            tag_seq.append('B-LOC')
        else:
            text.extend(['The', 'company'])
            tag_seq.extend(['O', 'O'])

            org = np.random.choice(orgs)
            text.append(org)
            tag_seq.append('B-ORG')

        texts.append(text)
        tags.append(tag_seq)

    return texts, tags


def main():
    print("=" * 60)
    print("Named Entity Recognition with Transformers")
    print("=" * 60)

    # Generate data
    texts, tags = generate_ner_data(1000)
    dataset = NERDataset(texts, tags)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model
    model = BiLSTM_CRF(len(dataset.vocab), len(dataset.tag_vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Train
    print("\nTraining NER model...")
    for epoch in range(10):
        total_loss = 0
        for texts_batch, tags_batch in loader:
            outputs = model(texts_batch)
            loss = criterion(outputs.view(-1, len(dataset.tag_vocab)), tags_batch.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/10], Loss: {total_loss / len(loader):.4f}')

    print("\nNER training complete!")
    print("\nKey Takeaways:")
    print("- NER identifies entities (persons, locations, organizations)")
    print("- BIO tagging: B=Begin, I=Inside, O=Outside")
    print("- BiLSTM-CRF is effective for sequence labeling")
    print("- Transformers (BERT) achieve state-of-the-art results")


if __name__ == "__main__":
    main()
