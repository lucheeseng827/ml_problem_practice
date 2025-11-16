"""
Transfer Learning with Pre-trained Models
==========================================
Category 21: Advanced ML - Using pre-trained models for custom tasks

Use cases: Fine-tuning pre-trained models, transfer learning, domain adaptation
Demonstrates: BERT, ResNet, model freezing, feature extraction, fine-tuning
"""

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoModel, AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
from torchvision import models, transforms
from PIL import Image


# =========================================================================
# PART 1: NLP - Fine-tuning BERT for Custom Classification
# =========================================================================

class BERTFineTuner:
    """Fine-tune BERT for custom text classification"""

    def __init__(self, model_name='bert-base-uncased', num_labels=3):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = None
        self.model = None

    def load_pretrained_model(self):
        """Load pre-trained BERT model"""
        print("=" * 70)
        print("LOADING PRE-TRAINED BERT MODEL")
        print("=" * 70)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print(f"✓ Loaded tokenizer: {self.model_name}")

        # Load pre-trained model with classification head
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )
        print(f"✓ Loaded model: {self.model_name}")
        print(f"✓ Model has {sum(p.numel() for p in self.model.parameters())} parameters")

        return self.model

    def demonstrate_feature_extraction(self):
        """Use BERT as feature extractor (frozen)"""
        print("\n" + "=" * 70)
        print("FEATURE EXTRACTION MODE (Frozen Base Model)")
        print("=" * 70)

        # Freeze all base model parameters
        for param in self.model.bert.parameters():
            param.requires_grad = False

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())

        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")
        print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")

        print("\n✓ Only classification head will be trained")
        print("✓ Base BERT weights remain frozen")
        print("✓ Faster training, less overfitting")

    def demonstrate_fine_tuning(self):
        """Fine-tune entire model (unfrozen)"""
        print("\n" + "=" * 70)
        print("FINE-TUNING MODE (Unfrozen Base Model)")
        print("=" * 70)

        # Unfreeze all parameters
        for param in self.model.parameters():
            param.requires_grad = True

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"Trainable parameters: {trainable_params:,}")
        print("✓ All model weights will be updated")
        print("✓ Slower training, more adaptation to task")
        print("✓ Better performance on custom domain")

    def demonstrate_layer_wise_lr(self):
        """Demonstrate discriminative learning rates"""
        print("\n" + "=" * 70)
        print("LAYER-WISE LEARNING RATES")
        print("=" * 70)

        # Different learning rates for different parts
        optimizer_config = [
            {'params': self.model.bert.embeddings.parameters(), 'lr': 1e-5},
            {'params': self.model.bert.encoder.parameters(), 'lr': 2e-5},
            {'params': self.model.classifier.parameters(), 'lr': 1e-4}
        ]

        print("Learning rate strategy:")
        print("  • Embeddings:     1e-5 (lowest - most general)")
        print("  • BERT encoder:   2e-5 (medium)")
        print("  • Classifier:     1e-4 (highest - most task-specific)")
        print()
        print("✓ Lower layers learn slowly (preserve pre-trained knowledge)")
        print("✓ Upper layers learn faster (adapt to new task)")


# =========================================================================
# PART 2: Computer Vision - Fine-tuning ResNet for Custom Images
# =========================================================================

class ResNetFineTuner:
    """Fine-tune ResNet for custom image classification"""

    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.model = None

    def load_pretrained_resnet(self):
        """Load pre-trained ResNet-50 from ImageNet"""
        print("\n" + "=" * 70)
        print("LOADING PRE-TRAINED RESNET-50")
        print("=" * 70)

        # Load pre-trained ResNet50
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        print(f"✓ Loaded ResNet-50 with ImageNet weights")
        print(f"✓ Original classes: 1000 (ImageNet)")
        print(f"✓ Model has {sum(p.numel() for p in self.model.parameters())} parameters")

        return self.model

    def replace_classification_head(self):
        """Replace final layer for custom number of classes"""
        print("\n" + "=" * 70)
        print("REPLACING CLASSIFICATION HEAD")
        print("=" * 70)

        # Get number of features in final layer
        num_features = self.model.fc.in_features
        print(f"Feature dimension: {num_features}")

        # Replace final fully connected layer
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, self.num_classes)
        )

        print(f"✓ Replaced final layer: {num_features} → {self.num_classes} classes")
        print("✓ Added dropout for regularization")

    def freeze_backbone(self):
        """Freeze backbone, only train classification head"""
        print("\n" + "=" * 70)
        print("FREEZING BACKBONE")
        print("=" * 70)

        # Freeze all layers except final fc
        for name, param in self.model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())

        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")
        print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")

        print("\n✓ Backbone frozen (feature extractor)")
        print("✓ Only classification head trainable")

    def unfreeze_last_n_layers(self, n=2):
        """Unfreeze last N residual blocks for fine-tuning"""
        print("\n" + "=" * 70)
        print(f"UNFREEZING LAST {n} RESIDUAL BLOCKS")
        print("=" * 70)

        # Unfreeze layer4 (last residual block) and fc
        layers_to_unfreeze = [f'layer{5-i}' for i in range(n)]
        layers_to_unfreeze.append('fc')

        for name, param in self.model.named_parameters():
            if any(layer in name for layer in layers_to_unfreeze):
                param.requires_grad = True

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())

        print(f"Unfrozen layers: {layers_to_unfreeze}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")

        print("\n✓ Gradual unfreezing strategy")
        print("✓ Balance between training speed and adaptation")


# =========================================================================
# PART 3: Unified Model Approach - Model Hub Integration
# =========================================================================

class UnifiedModelLoader:
    """Load and use models from HuggingFace Hub"""

    @staticmethod
    def load_any_model(model_name: str, task: str = 'feature-extraction'):
        """
        Load any model from HuggingFace Hub

        Supported tasks:
        - feature-extraction
        - text-classification
        - question-answering
        - image-classification
        - object-detection
        - etc.
        """
        print("=" * 70)
        print("UNIFIED MODEL HUB APPROACH")
        print("=" * 70)

        print(f"\nLoading: {model_name}")
        print(f"Task: {task}")

        # Example model names:
        models_by_task = {
            'text': [
                'bert-base-uncased',
                'roberta-base',
                'distilbert-base-uncased',
                'microsoft/deberta-v3-base',
                'google/electra-base-discriminator'
            ],
            'vision': [
                'microsoft/resnet-50',
                'google/vit-base-patch16-224',
                'facebook/deit-base-distilled-patch16-224',
                'microsoft/swin-base-patch4-window7-224'
            ],
            'multimodal': [
                'openai/clip-vit-base-patch32',
                'microsoft/layoutlm-base-uncased',
                'Salesforce/blip-image-captioning-base'
            ]
        }

        print("\n📚 Available pre-trained models:")
        for category, model_list in models_by_task.items():
            print(f"\n{category.upper()}:")
            for m in model_list:
                print(f"  • {m}")

        print("\n✓ All accessible via transformers library")
        print("✓ Automatic download and caching")
        print("✓ Unified API across all models")


# =========================================================================
# PART 4: Complete Model Lifecycle with Pre-trained Models
# =========================================================================

def demonstrate_complete_lifecycle():
    """Show complete lifecycle with transfer learning"""

    print("\n" + "=" * 70)
    print("COMPLETE MODEL LIFECYCLE WITH TRANSFER LEARNING")
    print("=" * 70)

    lifecycle_stages = {
        '1. Model Selection': {
            'description': 'Choose pre-trained model based on task',
            'actions': [
                'Identify task (classification, detection, etc.)',
                'Select model family (BERT, ResNet, etc.)',
                'Choose model size (base, large, etc.)',
                'Consider computational constraints'
            ]
        },
        '2. Model Loading': {
            'description': 'Load pre-trained weights',
            'actions': [
                'Download model from hub (HuggingFace, Torchvision)',
                'Initialize model architecture',
                'Load pre-trained weights',
                'Verify model loaded correctly'
            ]
        },
        '3. Model Adaptation': {
            'description': 'Adapt model to custom task',
            'actions': [
                'Replace classification head',
                'Add task-specific layers',
                'Configure layer freezing',
                'Set up discriminative learning rates'
            ]
        },
        '4. Training Strategy': {
            'description': 'Choose transfer learning approach',
            'options': {
                'Feature Extraction': 'Freeze base, train head only (fast)',
                'Fine-tuning': 'Unfreeze all, train everything (slower, better)',
                'Gradual Unfreezing': 'Progressively unfreeze layers (balanced)'
            }
        },
        '5. Training': {
            'description': 'Train adapted model',
            'actions': [
                'Prepare custom dataset',
                'Set hyperparameters (lr, batch_size, etc.)',
                'Train with validation monitoring',
                'Apply early stopping'
            ]
        },
        '6. Evaluation': {
            'description': 'Evaluate on test set',
            'actions': [
                'Calculate metrics on test set',
                'Compare with baseline',
                'Analyze failure cases',
                'Validate on deployment data'
            ]
        },
        '7. Deployment': {
            'description': 'Deploy to production',
            'actions': [
                'Export model (ONNX, TorchScript, SavedModel)',
                'Optimize for inference (quantization, pruning)',
                'Package with dependencies',
                'Deploy to endpoint (SageMaker, BentoML, etc.)'
            ]
        },
        '8. Monitoring': {
            'description': 'Monitor in production',
            'actions': [
                'Track prediction distribution',
                'Detect data drift',
                'Monitor performance metrics',
                'Trigger retraining when needed'
            ]
        }
    }

    for stage, details in lifecycle_stages.items():
        print(f"\n{'=' * 70}")
        print(f"{stage}: {details['description']}")
        print('=' * 70)

        if 'actions' in details:
            for action in details['actions']:
                print(f"  • {action}")
        elif 'options' in details:
            for option, desc in details['options'].items():
                print(f"  • {option}: {desc}")


# =========================================================================
# MAIN DEMONSTRATION
# =========================================================================

def main():
    print("=" * 70)
    print("TRANSFER LEARNING WITH PRE-TRAINED MODELS")
    print("=" * 70)

    # Part 1: BERT Fine-tuning
    print("\n" + "=" * 70)
    print("PART 1: NLP - BERT FOR TEXT CLASSIFICATION")
    print("=" * 70)

    bert_tuner = BERTFineTuner(num_labels=3)
    bert_tuner.load_pretrained_model()
    bert_tuner.demonstrate_feature_extraction()
    bert_tuner.demonstrate_fine_tuning()
    bert_tuner.demonstrate_layer_wise_lr()

    # Part 2: ResNet Fine-tuning
    print("\n" + "=" * 70)
    print("PART 2: COMPUTER VISION - RESNET FOR IMAGE CLASSIFICATION")
    print("=" * 70)

    resnet_tuner = ResNetFineTuner(num_classes=10)
    resnet_tuner.load_pretrained_resnet()
    resnet_tuner.replace_classification_head()
    resnet_tuner.freeze_backbone()
    resnet_tuner.unfreeze_last_n_layers(n=2)

    # Part 3: Unified Model Hub
    print("\n" + "=" * 70)
    print("PART 3: UNIFIED MODEL HUB")
    print("=" * 70)

    UnifiedModelLoader.load_any_model(
        'bert-base-uncased',
        task='text-classification'
    )

    # Part 4: Complete Lifecycle
    demonstrate_complete_lifecycle()

    # Summary
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
✓ PRE-TRAINED MODELS:
  • Start with models trained on massive datasets
  • Leverage learned features for custom tasks
  • Much faster than training from scratch
  • Better performance with less data

✓ TRANSFER LEARNING STRATEGIES:
  1. Feature Extraction: Freeze base, train head only
  2. Fine-tuning: Unfreeze all, train everything
  3. Gradual Unfreezing: Progressive layer unfreezing
  4. Discriminative LR: Different learning rates per layer

✓ MODEL SELECTION:
  • NLP: BERT, RoBERTa, DeBERTa, ELECTRA
  • Vision: ResNet, ViT, Swin, EfficientNet
  • Multi-modal: CLIP, BLIP, LayoutLM

✓ UNIFIED APPROACH:
  • HuggingFace Hub: 100,000+ pre-trained models
  • Torchvision: Standard vision models
  • TensorFlow Hub: TF models
  • All accessible via simple APIs

✓ PRODUCTION PIPELINE:
  1. Select pre-trained model
  2. Load and adapt to task
  3. Train with custom data
  4. Evaluate and validate
  5. Export and optimize
  6. Deploy to endpoint
  7. Monitor and retrain

✓ BENEFITS:
  • Faster development
  • Better performance
  • Less data required
  • Proven architectures
  • Community support
    """)

    print("\n" + "=" * 70)
    print("CODE EXAMPLES")
    print("=" * 70)
    print("""
# Load pre-trained BERT
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=3
)

# Load pre-trained ResNet
from torchvision import models

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False
model.fc.weight.requires_grad = True  # Only train final layer

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    """)


if __name__ == "__main__":
    main()
