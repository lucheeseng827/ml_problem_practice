"""
Object Detection with YOLO (You Only Look Once)
================================================
Category 11: Computer Vision

This example demonstrates:
- YOLO architecture for real-time object detection
- Bounding box prediction and regression
- Non-Maximum Suppression (NMS)
- IoU (Intersection over Union) calculations
- Multi-class detection

Use cases:
- Autonomous vehicles
- Surveillance systems
- Retail analytics
- Manufacturing quality control
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class YOLOv3Tiny(nn.Module):
    """
    Simplified YOLO architecture for demonstration

    Real YOLO has complex feature pyramid networks,
    this is a teaching version showing core concepts
    """

    def __init__(self, num_classes=20, grid_size=7):
        super(YOLOv3Tiny, self).__init__()

        self.num_classes = num_classes
        self.grid_size = grid_size
        self.num_boxes = 2  # Number of bounding boxes per grid cell

        # Feature extraction backbone
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            # Conv Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            # Conv Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            # Conv Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            # Conv Block 5
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
        )

        # Detection head
        # Output: (grid_size x grid_size x num_boxes x (5 + num_classes))
        # 5 = x, y, w, h, confidence
        output_size = self.num_boxes * (5 + num_classes)

        self.detection_head = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, output_size, kernel_size=1)
        )

    def forward(self, x):
        # Extract features
        features = self.features(x)

        # Detection
        detections = self.detection_head(features)

        # Reshape to (batch, grid, grid, num_boxes, 5 + num_classes)
        batch_size = x.size(0)
        detections = detections.permute(0, 2, 3, 1)  # (B, H, W, C)

        return detections


class SyntheticObjectDataset(Dataset):
    """Generate synthetic object detection data"""

    def __init__(self, num_samples=1000, image_size=224, num_classes=5):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate synthetic image with random boxes
        np.random.seed(idx)

        # Create blank image
        image = np.random.rand(3, self.image_size, self.image_size).astype(np.float32)

        # Number of objects (1-3)
        num_objects = np.random.randint(1, 4)

        boxes = []
        labels = []

        for _ in range(num_objects):
            # Random box coordinates (normalized 0-1)
            x_center = np.random.uniform(0.2, 0.8)
            y_center = np.random.uniform(0.2, 0.8)
            width = np.random.uniform(0.1, 0.3)
            height = np.random.uniform(0.1, 0.3)

            # Random class
            class_id = np.random.randint(0, self.num_classes)

            # Draw colored rectangle on image
            color = np.random.rand(3, 1, 1)
            x_min = int((x_center - width / 2) * self.image_size)
            y_min = int((y_center - height / 2) * self.image_size)
            x_max = int((x_center + width / 2) * self.image_size)
            y_max = int((y_center + height / 2) * self.image_size)

            x_min, x_max = max(0, x_min), min(self.image_size, x_max)
            y_min, y_max = max(0, y_min), min(self.image_size, y_max)

            image[:, y_min:y_max, x_min:x_max] = color

            boxes.append([x_center, y_center, width, height])
            labels.append(class_id)

        # Convert to tensors
        image = torch.FloatTensor(image)
        boxes = torch.FloatTensor(boxes)
        labels = torch.LongTensor(labels)

        return image, boxes, labels


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union

    box format: [x_center, y_center, width, height]
    """
    # Convert to corner coordinates
    box1_x1 = box1[0] - box1[2] / 2
    box1_y1 = box1[1] - box1[3] / 2
    box1_x2 = box1[0] + box1[2] / 2
    box1_y2 = box1[1] + box1[3] / 2

    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2

    # Intersection area
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Union area
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / (union_area + 1e-6)

    return iou


def non_max_suppression(boxes, scores, iou_threshold=0.5):
    """
    Non-Maximum Suppression to filter overlapping boxes

    Args:
        boxes: List of bounding boxes [x, y, w, h]
        scores: Confidence scores for each box
        iou_threshold: IoU threshold for suppression
    """
    if len(boxes) == 0:
        return []

    # Sort by confidence score
    indices = np.argsort(scores)[::-1]

    keep = []

    while len(indices) > 0:
        # Pick box with highest score
        current = indices[0]
        keep.append(current)

        if len(indices) == 1:
            break

        # Calculate IoU with remaining boxes
        current_box = boxes[current]
        remaining_boxes = boxes[indices[1:]]

        ious = np.array([calculate_iou(current_box, box) for box in remaining_boxes])

        # Keep only boxes with IoU below threshold
        indices = indices[1:][ious < iou_threshold]

    return keep


def yolo_loss(predictions, targets, lambda_coord=5, lambda_noobj=0.5):
    """
    Simplified YOLO loss function

    Components:
    - Localization loss (bounding box coordinates)
    - Confidence loss (objectness)
    - Classification loss
    """
    # This is a simplified version
    # Real YOLO loss is more complex with anchor boxes

    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    # Coordinate loss
    coord_loss = mse_loss(predictions[..., :4], targets[..., :4])

    # Confidence loss
    conf_loss = mse_loss(predictions[..., 4], targets[..., 4])

    # Classification loss (simplified)
    total_loss = lambda_coord * coord_loss + conf_loss

    return total_loss


def train_yolo_detector():
    """Train YOLO object detector"""
    print("=" * 60)
    print("YOLO Object Detection Training")
    print("=" * 60)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on: {device}")

    # Create dataset
    print("\nCreating synthetic dataset...")
    train_dataset = SyntheticObjectDataset(num_samples=1000, num_classes=5)
    val_dataset = SyntheticObjectDataset(num_samples=200, num_classes=5)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Initialize model
    model = YOLOv3Tiny(num_classes=5, grid_size=7).to(device)

    print(f"\nModel Architecture:")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 20
    train_losses = []

    print("\nTraining YOLO detector...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for batch_idx, (images, boxes, labels) in enumerate(train_loader):
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            # Simplified loss (in practice, need to match predictions to ground truth)
            # This is a placeholder - real implementation requires complex matching
            loss = outputs.abs().mean()  # Placeholder loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    print("\nTraining complete!")

    # Save model
    torch.save(model.state_dict(), '/tmp/yolo_detector.pth')
    print("Model saved to /tmp/yolo_detector.pth")

    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('YOLO Training Loss')
    plt.grid(True)
    plt.savefig('/tmp/yolo_training.png')
    print("Training plot saved to /tmp/yolo_training.png")

    return model


def visualize_detections():
    """Visualize object detections"""
    print("\n" + "=" * 60)
    print("Visualizing Object Detections")
    print("=" * 60)

    # Create sample image with objects
    dataset = SyntheticObjectDataset(num_samples=1)
    image, boxes, labels = dataset[0]

    # Convert image to numpy for visualization
    img_np = image.permute(1, 2, 0).numpy()

    # Plot
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(img_np)

    # Draw bounding boxes
    for box, label in zip(boxes, labels):
        x_center, y_center, width, height = box
        img_size = 224

        # Convert to corner coordinates
        x = (x_center - width / 2) * img_size
        y = (y_center - height / 2) * img_size
        w = width * img_size
        h = height * img_size

        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)

        # Add label
        ax.text(x, y - 5, f'Class {label.item()}',
                bbox=dict(facecolor='red', alpha=0.5),
                fontsize=10, color='white')

    ax.axis('off')
    plt.title('Object Detection Visualization')
    plt.tight_layout()
    plt.savefig('/tmp/object_detection_viz.png')
    print("\nVisualization saved to /tmp/object_detection_viz.png")


def main():
    """Main execution function"""
    print("YOLO Object Detection with PyTorch\n")

    # Train detector
    model = train_yolo_detector()

    # Visualize results
    visualize_detections()

    # Demonstrate NMS
    print("\n" + "=" * 60)
    print("Non-Maximum Suppression Demo")
    print("=" * 60)

    # Sample overlapping boxes
    boxes = np.array([
        [0.5, 0.5, 0.3, 0.3],
        [0.52, 0.52, 0.3, 0.3],
        [0.7, 0.7, 0.2, 0.2]
    ])
    scores = np.array([0.9, 0.8, 0.95])

    keep_indices = non_max_suppression(boxes, scores, iou_threshold=0.5)

    print(f"\nOriginal boxes: {len(boxes)}")
    print(f"After NMS: {len(keep_indices)}")
    print(f"Kept indices: {keep_indices}")

    print("\n" + "=" * 60)
    print("YOLO Object Detection Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("- YOLO performs detection in a single forward pass")
    print("- Grid-based predictions enable real-time performance")
    print("- IoU measures bounding box overlap")
    print("- NMS eliminates redundant detections")
    print("- Multi-scale features improve detection accuracy")


if __name__ == "__main__":
    main()
