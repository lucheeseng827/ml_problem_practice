"""
Stable Diffusion - Text-to-Image Generation
============================================
Category 18: Multi-Modal - Generate images from text

Use cases: Art generation, design, content creation
"""

import numpy as np


class SimpleStableDiffusion:
    """Simplified Stable Diffusion concepts"""
    
    def __init__(self, image_size=64):
        self.image_size = image_size
    
    def generate(self, text_prompt, num_steps=20):
        """Simulate diffusion process"""
        print(f"\nGenerating image from: '{text_prompt}'")
        
        # Start with random noise
        image = np.random.randn(self.image_size, self.image_size, 3)
        
        # Iterative denoising (simplified)
        for step in range(num_steps):
            # In reality, uses U-Net to predict noise
            noise_pred = np.random.randn(*image.shape) * 0.1
            image = image - noise_pred
            
            if (step + 1) % 5 == 0:
                print(f"  Step {step + 1}/{num_steps}")
        
        # Normalize to [0, 1]
        image = (image - image.min()) / (image.max() - image.min())
        
        return image


def main():
    print("=" * 60)
    print("Stable Diffusion - Text-to-Image")
    print("=" * 60)
    
    model = SimpleStableDiffusion(image_size=64)
    
    prompts = [
        "A serene mountain landscape at sunset",
        "A futuristic city with flying cars",
        "An adorable robot playing with a cat"
    ]
    
    for prompt in prompts:
        image = model.generate(prompt, num_steps=20)
        print(f"Generated image shape: {image.shape}\n")
    
    print("=" * 60)
    print("Key Takeaways:")
    print("- Stable Diffusion uses latent diffusion models")
    print("- Text encoded with CLIP, image decoded with U-Net")
    print("- Iterative denoising process")
    print("- Enables creative AI applications")
    print("- Open source, runs on consumer GPUs")


if __name__ == "__main__":
    main()
