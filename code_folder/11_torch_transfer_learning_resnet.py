"""
Transfer Learning with ResNet
==============================
Category 11: Computer Vision

This example demonstrates:
- Transfer learning from pretrained models
- Fine-tuning strategies
- Feature extraction vs full fine-tuning
- Data augmentation techniques
- Learning rate scheduling

Use cases:
- Custom image classification with limited data
- Domain adaptation
- Quick prototyping
- Medical imaging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns


class CustomImageDataset(Dataset):
    """Custom dataset for transfer learning"""

    def __init__(self, num_samples=1000, num_classes=10, img_size=224, transform=None):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.img_size = img_size
        self.transform = transform

        # Generate synthetic data
        np.random.seed(42)
        self.images = []
        self.labels = []

        for _ in range(num_samples):
            # Random class
            label = np.random.randint(0, num_classes)

            # Generate image with class-specific pattern
            img = np.random.rand(3, img_size, img_size) * 0.5

            # Add class-specific pattern
            pattern = np.sin(2 * np.pi * label / num_classes)
            img += pattern * 0.3

            self.images.append(img.astype(np.float32))
            self.labels.append(label)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.FloatTensor(self.images[idx])
        label = torch.LongTensor([self.labels[idx]])[0]

        if self.transform:
            image = self.transform(image)

        return image, label


def create_resnet_model(num_classes, pretrained=True, feature_extract=False):
    """
    Create ResNet model for transfer learning

    Args:
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        feature_extract: If True, only train final layer
    """
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=pretrained)

    if feature_extract:
        # Freeze all layers except final
        for param in model.parameters():
            param.requires_grad = False

    # Modify final layer for our num_classes
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )

    return model


def create_resnet50_model(num_classes):
    """Create deeper ResNet-50 model"""
    model = models.resnet50(pretrained=True)

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

    return model


def get_data_augmentation():
    """Data augmentation transforms"""
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    predictions = []
    targets = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Track predictions
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())
        targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(targets, predictions)

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    predictions = []
    targets = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    val_loss = running_loss / len(dataloader)
    val_acc = accuracy_score(targets, predictions)

    return val_loss, val_acc, predictions, targets


def feature_extraction_training():
    """Transfer learning with feature extraction (frozen backbone)"""
    print("=" * 60)
    print("Transfer Learning - Feature Extraction")
    print("=" * 60)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on: {device}")

    # Create datasets
    print("\nCreating datasets...")
    train_transform, val_transform = get_data_augmentation()

    train_dataset = CustomImageDataset(num_samples=800, num_classes=10, transform=train_transform)
    val_dataset = CustomImageDataset(num_samples=200, num_classes=10, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create model (frozen backbone)
    print("\nCreating ResNet-18 with frozen backbone...")
    model = create_resnet_model(num_classes=10, pretrained=True, feature_extract=True)
    model = model.to(device)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    # Training loop
    num_epochs = 20
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print("\nTraining (feature extraction mode)...")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}]')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    print("\nFeature extraction training complete!")

    return model, train_losses, val_losses, train_accs, val_accs


def fine_tuning_training():
    """Transfer learning with fine-tuning (unfrozen layers)"""
    print("\n" + "=" * 60)
    print("Transfer Learning - Fine-Tuning")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create datasets
    train_transform, val_transform = get_data_augmentation()
    train_dataset = CustomImageDataset(num_samples=800, num_classes=10, transform=train_transform)
    val_dataset = CustomImageDataset(num_samples=200, num_classes=10, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create model (all layers trainable)
    print("\nCreating ResNet-18 with trainable backbone...")
    model = create_resnet_model(num_classes=10, pretrained=True, feature_extract=False)
    model = model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    # Discriminative learning rates
    # Lower LR for pretrained layers, higher for new layers
    optimizer = optim.Adam([
        {'params': model.layer1.parameters(), 'lr': 1e-5},
        {'params': model.layer2.parameters(), 'lr': 1e-5},
        {'params': model.layer3.parameters(), 'lr': 1e-4},
        {'params': model.layer4.parameters(), 'lr': 1e-4},
        {'params': model.fc.parameters(), 'lr': 1e-3}
    ])

    criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # Training
    num_epochs = 20
    train_losses, val_losses = [], []

    print("\nFine-tuning all layers...")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, preds, targets = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Update learning rate
        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}]')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    # Confusion matrix
    cm = confusion_matrix(targets, preds)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Fine-Tuned ResNet')
    plt.tight_layout()
    plt.savefig('/tmp/resnet_confusion_matrix.png')
    print("\nConfusion matrix saved to /tmp/resnet_confusion_matrix.png")

    # Save model
    torch.save(model.state_dict(), '/tmp/resnet_finetuned.pth')
    print("Model saved to /tmp/resnet_finetuned.pth")

    return model, train_losses, val_losses


def visualize_training_comparison(fe_train, fe_val, ft_train, ft_val):
    """Compare feature extraction vs fine-tuning"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Feature Extraction
    axes[0].plot(fe_train, label='Train Loss')
    axes[0].plot(fe_val, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Feature Extraction (Frozen Backbone)')
    axes[0].legend()
    axes[0].grid(True)

    # Fine-Tuning
    axes[1].plot(ft_train, label='Train Loss')
    axes[1].plot(ft_val, label='Val Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Fine-Tuning (Unfrozen Backbone)')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('/tmp/transfer_learning_comparison.png')
    print("\nComparison plot saved to /tmp/transfer_learning_comparison.png")


def main():
    """Main execution function"""
    print("Transfer Learning with ResNet\n")

    # Approach 1: Feature Extraction
    fe_model, fe_train_loss, fe_val_loss, fe_train_acc, fe_val_acc = feature_extraction_training()

    # Approach 2: Fine-Tuning
    ft_model, ft_train_loss, ft_val_loss = fine_tuning_training()

    # Visualize comparison
    visualize_training_comparison(fe_train_loss, fe_val_loss, ft_train_loss, ft_val_loss)

    print("\n" + "=" * 60)
    print("Transfer Learning Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("- Feature extraction: fast, good for small datasets")
    print("- Fine-tuning: better performance, needs more data")
    print("- Discriminative LR: lower for pretrained, higher for new layers")
    print("- Data augmentation helps prevent overfitting")
    print("- Pretrained models provide excellent initialization")
    print("- ImageNet features transfer well to many domains")


if __name__ == "__main__":
    main()
