"""
Data Augmentation Script
Generates augmented images from the existing dataset to increase training data.
Each original image produces multiple augmented versions using various transformations.
"""

import cv2
import numpy as np
import os
import random

# Augmentation parameters
AUGMENTATIONS_PER_IMAGE = 8  # Number of augmented images per original


def random_rotation(img):
    """Rotate image by a random angle between -30 and 30 degrees."""
    angle = random.uniform(-30, 30)
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)


def random_flip(img):
    """Randomly flip image horizontally and/or vertically."""
    flip_code = random.choice([-1, 0, 1])  # both, vertical, horizontal
    return cv2.flip(img, flip_code)


def random_brightness(img):
    """Adjust brightness randomly."""
    factor = random.uniform(0.6, 1.4)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def random_contrast(img):
    """Adjust contrast randomly."""
    factor = random.uniform(0.7, 1.3)
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    return np.clip((img - mean) * factor + mean, 0, 255).astype(np.uint8)


def random_zoom(img):
    """Randomly zoom into the image."""
    h, w = img.shape[:2]
    scale = random.uniform(0.7, 1.0)
    new_h, new_w = int(h * scale), int(w * scale)
    y_start = random.randint(0, h - new_h)
    x_start = random.randint(0, w - new_w)
    cropped = img[y_start:y_start + new_h, x_start:x_start + new_w]
    return cv2.resize(cropped, (w, h))


def random_noise(img):
    """Add random Gaussian noise."""
    noise = np.random.normal(0, 15, img.shape).astype(np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


def random_blur(img):
    """Apply random Gaussian blur."""
    ksize = random.choice([3, 5])
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def random_shift(img):
    """Randomly shift the image."""
    h, w = img.shape[:2]
    tx = random.randint(-int(w * 0.15), int(w * 0.15))
    ty = random.randint(-int(h * 0.15), int(h * 0.15))
    matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)


def augment_image(img):
    """Apply a random combination of augmentations to an image."""
    augmentations = [
        random_rotation,
        random_flip,
        random_brightness,
        random_contrast,
        random_zoom,
        random_noise,
        random_blur,
        random_shift,
    ]

    # Apply 2-4 random augmentations
    num_augs = random.randint(2, 4)
    selected = random.sample(augmentations, num_augs)

    result = img.copy()
    for aug_fn in selected:
        result = aug_fn(result)

    return result


def augment_dataset(dataset_dir):
    """Augment all images in the dataset directory."""

    classes = ["bacterial_leaf_blight", "brown_spot", "healthy", "leaf_smut"]

    total_original = 0
    total_augmented = 0

    for class_name in classes:
        class_dir = os.path.join(dataset_dir, class_name)

        if not os.path.isdir(class_dir):
            print(f"  [SKIP] {class_name} — folder not found")
            continue

        # Get only original images (not previously augmented ones)
        images = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
            and not f.startswith('aug_')
        ]

        original_count = len(images)
        total_original += original_count
        aug_count = 0

        print(f"\n  [{class_name}] — {original_count} original images")

        for img_name in images:
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)

            if img is None:
                print(f"    [WARN] Could not read: {img_name}")
                continue

            # Generate augmented versions
            for i in range(AUGMENTATIONS_PER_IMAGE):
                augmented = augment_image(img)
                base_name = os.path.splitext(img_name)[0]
                aug_filename = f"aug_{base_name}_{i}.jpg"
                aug_path = os.path.join(class_dir, aug_filename)
                cv2.imwrite(aug_path, augmented)
                aug_count += 1

        total_augmented += aug_count
        print(f"    Generated {aug_count} augmented images")
        print(f"    Total in class: {original_count + aug_count} images")

    print(f"\n{'='*50}")
    print(f"  SUMMARY")
    print(f"{'='*50}")
    print(f"  Original images:  {total_original}")
    print(f"  Augmented images: {total_augmented}")
    print(f"  Total images:     {total_original + total_augmented}")
    print(f"{'='*50}")


if __name__ == "__main__":
    print("=" * 50)
    print("  PADDY LEAF DISEASE — DATA AUGMENTATION")
    print("=" * 50)

    dataset_dir = "dataset"
    augment_dataset(dataset_dir)

    print("\n  Data augmentation complete!")
