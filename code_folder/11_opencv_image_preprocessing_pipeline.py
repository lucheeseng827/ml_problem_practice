"""
Image Preprocessing Pipeline with OpenCV
=========================================
Category 11: Computer Vision

This example demonstrates:
- Image loading and color space conversions
- Resizing, cropping, and geometric transformations
- Filtering and noise reduction
- Histogram equalization and contrast enhancement
- Edge detection and feature extraction
- Complete preprocessing pipeline for ML

Use cases:
- Data preparation for computer vision models
- Image quality enhancement
- Feature engineering
- Real-time video processing
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def create_sample_images():
    """Create sample images for demonstration"""
    # Create various test images
    images = {}

    # 1. Gradient image
    gradient = np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (512, 1))
    images['gradient'] = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)

    # 2. Checkerboard pattern
    checker = np.zeros((512, 512), dtype=np.uint8)
    checker[::64, :] = 255
    checker[:, ::64] = 255
    images['checker'] = cv2.cvtColor(checker, cv2.COLOR_GRAY2BGR)

    # 3. Noisy image
    noisy = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    images['noisy'] = noisy

    # 4. Shape image
    shapes = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.circle(shapes, (150, 150), 50, (255, 0, 0), -1)
    cv2.rectangle(shapes, (300, 100), (400, 200), (0, 255, 0), -1)
    cv2.ellipse(shapes, (256, 350), (100, 50), 0, 0, 360, (0, 0, 255), -1)
    images['shapes'] = shapes

    # 5. Low contrast image
    low_contrast = np.random.randint(100, 155, (512, 512, 3), dtype=np.uint8)
    images['low_contrast'] = low_contrast

    return images


def color_space_conversions(image):
    """Demonstrate color space conversions"""
    print("=" * 60)
    print("Color Space Conversions")
    print("=" * 60)

    conversions = {}

    # RGB to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    conversions['grayscale'] = gray

    # RGB to HSV (Hue, Saturation, Value)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    conversions['hsv'] = hsv

    # RGB to LAB (Lightness, A, B)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    conversions['lab'] = lab

    # RGB to YCrCb
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    conversions['ycrcb'] = ycrcb

    print("\nColor space conversions:")
    for name, img in conversions.items():
        print(f"  {name}: shape {img.shape}")

    return conversions


def geometric_transformations(image):
    """Apply geometric transformations"""
    print("\n" + "=" * 60)
    print("Geometric Transformations")
    print("=" * 60)

    h, w = image.shape[:2]
    transforms = {}

    # Resize
    resized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
    transforms['resized'] = resized
    print(f"\nResized: {image.shape} -> {resized.shape}")

    # Rotate
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
    transforms['rotated'] = rotated
    print(f"Rotated: 45 degrees")

    # Flip
    flipped_h = cv2.flip(image, 1)  # Horizontal
    flipped_v = cv2.flip(image, 0)  # Vertical
    transforms['flipped_horizontal'] = flipped_h
    transforms['flipped_vertical'] = flipped_v
    print(f"Flipped: horizontal and vertical")

    # Crop
    crop_h, crop_w = h // 2, w // 2
    cropped = image[crop_h-128:crop_h+128, crop_w-128:crop_w+128]
    transforms['cropped'] = cropped
    print(f"Cropped: center 256x256")

    # Affine transformation
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    affine_matrix = cv2.getAffineTransform(pts1, pts2)
    affine = cv2.warpAffine(image, affine_matrix, (w, h))
    transforms['affine'] = affine
    print(f"Affine transformation applied")

    # Perspective transformation
    pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
    pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
    perspective_matrix = cv2.getPerspectiveTransform(pts1, pts2)
    perspective = cv2.warpPerspective(image, perspective_matrix, (300, 300))
    transforms['perspective'] = perspective
    print(f"Perspective transformation applied")

    return transforms


def filtering_and_smoothing(image):
    """Apply various filtering techniques"""
    print("\n" + "=" * 60)
    print("Filtering and Smoothing")
    print("=" * 60)

    filters = {}

    # Gaussian Blur
    gaussian = cv2.GaussianBlur(image, (5, 5), 0)
    filters['gaussian_blur'] = gaussian
    print("\nGaussian Blur: kernel 5x5")

    # Median Blur (good for salt-and-pepper noise)
    median = cv2.medianBlur(image, 5)
    filters['median_blur'] = median
    print("Median Blur: kernel 5")

    # Bilateral Filter (edge-preserving)
    bilateral = cv2.bilateralFilter(image, 9, 75, 75)
    filters['bilateral'] = bilateral
    print("Bilateral Filter: d=9, sigmaColor=75, sigmaSpace=75")

    # Box Filter
    box = cv2.boxFilter(image, -1, (5, 5))
    filters['box_filter'] = box
    print("Box Filter: kernel 5x5")

    # Morphological operations (for binary images)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(binary, kernel, iterations=1)
    dilation = cv2.dilate(binary, kernel, iterations=1)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    filters['erosion'] = erosion
    filters['dilation'] = dilation
    filters['opening'] = opening
    filters['closing'] = closing

    print("Morphological operations applied")

    return filters


def contrast_enhancement(image):
    """Enhance image contrast"""
    print("\n" + "=" * 60)
    print("Contrast Enhancement")
    print("=" * 60)

    enhancements = {}

    # Histogram Equalization (grayscale)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    enhancements['histogram_equalization'] = equalized
    print("\nHistogram equalization applied")

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_applied = clahe.apply(gray)
    enhancements['clahe'] = clahe_applied
    print("CLAHE applied: clipLimit=2.0, tileGridSize=8x8")

    # Gamma Correction
    def adjust_gamma(image, gamma=1.0):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    gamma_corrected = adjust_gamma(image, gamma=1.5)
    enhancements['gamma_correction'] = gamma_corrected
    print(f"Gamma correction: gamma=1.5")

    # Adaptive Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    enhancements['adaptive_threshold'] = adaptive_thresh
    print("Adaptive thresholding applied")

    return enhancements


def edge_detection(image):
    """Apply edge detection algorithms"""
    print("\n" + "=" * 60)
    print("Edge Detection")
    print("=" * 60)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = {}

    # Sobel
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)
    edges['sobel'] = sobel
    print("\nSobel edge detection applied")

    # Scharr (more accurate than Sobel)
    scharrx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    scharry = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    scharr = np.sqrt(scharrx**2 + scharry**2).astype(np.uint8)
    edges['scharr'] = scharr
    print("Scharr edge detection applied")

    # Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    edges['laplacian'] = laplacian
    print("Laplacian edge detection applied")

    # Canny
    canny = cv2.Canny(gray, 100, 200)
    edges['canny'] = canny
    print("Canny edge detection: threshold1=100, threshold2=200")

    return edges


def feature_extraction(image):
    """Extract features from image"""
    print("\n" + "=" * 60)
    print("Feature Extraction")
    print("=" * 60)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = {}

    # Harris Corner Detection
    gray_float = np.float32(gray)
    harris = cv2.cornerHarris(gray_float, 2, 3, 0.04)
    harris_dilated = cv2.dilate(harris, None)
    features['harris_corners'] = harris_dilated
    print("\nHarris corner detection applied")

    # Shi-Tomasi Corner Detection
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    features['shi_tomasi_corners'] = corners
    print(f"Shi-Tomasi: detected {len(corners) if corners is not None else 0} corners")

    # SIFT (Scale-Invariant Feature Transform)
    try:
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        sift_image = cv2.drawKeypoints(gray, keypoints, None)
        features['sift'] = sift_image
        features['sift_keypoints'] = keypoints
        print(f"SIFT: detected {len(keypoints)} keypoints")
    except:
        print("SIFT not available (may require opencv-contrib-python)")

    # ORB (Oriented FAST and Rotated BRIEF)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    orb_image = cv2.drawKeypoints(gray, keypoints, None, color=(0, 255, 0))
    features['orb'] = orb_image
    features['orb_keypoints'] = keypoints
    print(f"ORB: detected {len(keypoints)} keypoints")

    return features


def complete_preprocessing_pipeline(image):
    """Complete preprocessing pipeline for ML"""
    print("\n" + "=" * 60)
    print("Complete Preprocessing Pipeline")
    print("=" * 60)

    pipeline_steps = {}

    # Step 1: Resize to standard size
    resized = cv2.resize(image, (224, 224))
    pipeline_steps['1_resized'] = resized
    print("\n1. Resized to 224x224")

    # Step 2: Denoise
    denoised = cv2.fastNlMeansDenoisingColored(resized, None, 10, 10, 7, 21)
    pipeline_steps['2_denoised'] = denoised
    print("2. Denoising applied")

    # Step 3: Convert to LAB color space
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    pipeline_steps['3_lab'] = lab

    # Step 4: Apply CLAHE to L channel
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge([l_clahe, a, b])
    pipeline_steps['4_lab_clahe'] = lab_clahe

    # Step 5: Convert back to BGR
    enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    pipeline_steps['5_enhanced'] = enhanced
    print("3. Contrast enhanced with CLAHE")

    # Step 6: Normalize to [0, 1]
    normalized = enhanced.astype(np.float32) / 255.0
    pipeline_steps['6_normalized'] = normalized
    print("4. Normalized to [0, 1]")

    # Step 7: Apply augmentations (example)
    # Random horizontal flip
    if np.random.rand() > 0.5:
        augmented = cv2.flip(normalized, 1)
    else:
        augmented = normalized
    pipeline_steps['7_augmented'] = augmented
    print("5. Random augmentation applied")

    print("\nPipeline complete! Ready for ML model input.")

    return pipeline_steps


def visualize_results(images_dict, title, filename):
    """Visualize processing results"""
    n_images = len(images_dict)
    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n_images > 1 else [axes]

    for idx, (name, img) in enumerate(images_dict.items()):
        if idx < len(axes):
            if len(img.shape) == 2:
                axes[idx].imshow(img, cmap='gray')
            else:
                # Convert BGR to RGB for matplotlib
                if img.dtype == np.uint8:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img_rgb = img
                axes[idx].imshow(img_rgb)
            axes[idx].set_title(name.replace('_', ' ').title())
            axes[idx].axis('off')

    # Hide extra subplots
    for idx in range(len(images_dict), len(axes)):
        axes[idx].axis('off')

    plt.suptitle(title, fontsize=16, y=1.0)
    plt.tight_layout()
    plt.savefig(f'/tmp/{filename}')
    print(f"\nVisualization saved to /tmp/{filename}")


def main():
    """Main execution function"""
    print("OpenCV Image Preprocessing Pipeline\n")

    # Create sample images
    print("Creating sample images...")
    sample_images = create_sample_images()

    # Use shapes image for demonstrations
    test_image = sample_images['shapes']

    # 1. Color space conversions
    color_spaces = color_space_conversions(test_image)
    visualize_results(color_spaces, 'Color Space Conversions', 'color_spaces.png')

    # 2. Geometric transformations
    transforms = geometric_transformations(test_image)
    visualize_results(
        {k: v for k, v in list(transforms.items())[:6]},
        'Geometric Transformations',
        'geometric_transforms.png'
    )

    # 3. Filtering
    filters = filtering_and_smoothing(sample_images['noisy'])
    visualize_results(
        {k: v for k, v in list(filters.items())[:6]},
        'Filtering and Smoothing',
        'filters.png'
    )

    # 4. Contrast enhancement
    enhancements = contrast_enhancement(sample_images['low_contrast'])
    visualize_results(enhancements, 'Contrast Enhancement', 'contrast.png')

    # 5. Edge detection
    edges = edge_detection(test_image)
    visualize_results(edges, 'Edge Detection', 'edges.png')

    # 6. Feature extraction
    features = feature_extraction(test_image)
    visualize_results(
        {k: v for k, v in features.items() if not k.endswith('_keypoints')},
        'Feature Extraction',
        'features.png'
    )

    # 7. Complete pipeline
    pipeline = complete_preprocessing_pipeline(test_image)

    print("\n" + "=" * 60)
    print("Image Preprocessing Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("- Color space conversion aids specific tasks")
    print("- Geometric transforms for data augmentation")
    print("- Filtering removes noise and smooths images")
    print("- Contrast enhancement improves visibility")
    print("- Edge detection finds boundaries")
    print("- Feature extraction for matching/tracking")
    print("- Pipeline ensures consistent preprocessing")


if __name__ == "__main__":
    main()
