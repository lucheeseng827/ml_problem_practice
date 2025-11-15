"""
CLIP - Image-Text Retrieval
============================
Category 18: Multi-Modal - Connecting vision and language

Use cases: Image search, zero-shot classification
"""

import numpy as np
import torch
import torch.nn as nn


class SimpleCLIP(nn.Module):
    def __init__(self, image_dim=512, text_dim=512, embed_dim=256):
        super(SimpleCLIP, self).__init__()
        self.image_encoder = nn.Linear(image_dim, embed_dim)
        self.text_encoder = nn.Linear(text_dim, embed_dim)
    
    def forward(self, images, texts):
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(texts)
        
        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity
        similarity = image_features @ text_features.T
        
        return similarity


def main():
    print("=" * 60)
    print("CLIP - Image-Text Retrieval")
    print("=" * 60)
    
    # Simulated image and text features
    images = torch.randn(32, 512)  # Batch of 32 images
    texts = torch.randn(32, 512)    # Batch of 32 text descriptions
    
    model = SimpleCLIP()
    
    # Compute similarity
    similarity = model(images, texts)
    
    print(f"\nSimilarity matrix shape: {similarity.shape}")
    print(f"Top similarity score: {similarity.max().item():.4f}")
    
    # Find best matches
    best_text_for_image0 = similarity[0].argmax().item()
    print(f"\nBest text match for image 0: text {best_text_for_image0}")
    
    print("\nKey Takeaways:")
    print("- CLIP learns joint vision-language embeddings")
    print("- Trained on 400M image-text pairs")
    print("- Enables zero-shot image classification")
    print("- Powers image search and generation")


if __name__ == "__main__":
    main()
