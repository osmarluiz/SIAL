"""
DataLoader creation module for training.
"""
from pathlib import Path
import numpy as np
import imageio
import torch
import torch.utils.data as data
from torchvision import transforms
import random


class RandomRotation90:
    """
    Randomly rotate the image by 0, 90, 180, or 270 degrees.

    Perfect for aerial imagery where there is no canonical "up" direction.
    Uses torch.rot90 for lossless rotation (no interpolation artifacts).
    """
    def __call__(self, x):
        # x is a tensor of shape (C, H, W)
        k = random.randint(0, 3)  # 0, 1, 2, or 3 (number of 90° rotations)
        if k > 0:
            x = torch.rot90(x, k, dims=[1, 2])
        return x


class CustomDataset(data.Dataset):
    """
    Custom dataset for semantic segmentation.

    Uses random seed synchronization to apply same augmentation to both image and mask.
    - Uses imageio for loading
    - Clips mask values > ignore_index
    - Supports label transforms with synchronized randomness
    """
    def __init__(self, image_paths, mask_paths, ignore_index, transform=None, transform_label=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.ignore_index = ignore_index
        self.transform = transform
        self.transform_label = transform_label

    def __getitem__(self, index):
        # Load image (using imageio like old notebook)
        image = imageio.imread(self.image_paths[index])
        image = np.asarray(image, dtype='float32')

        # Load mask (using imageio like old notebook)
        mask = imageio.imread(self.mask_paths[index])
        mask = np.asarray(mask, dtype='int64')

        # Clip values > ignore_index to ignore_index (like old notebook)
        mask[mask > self.ignore_index] = self.ignore_index

        # Set random seed to ensure same augmentation for image and mask
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)

        # Apply transform to image (includes ToTensor and optional augmentation)
        if self.transform is not None:
            image = self.transform(image)

        # Reset to same seed for mask transform
        random.seed(seed)
        torch.manual_seed(seed)

        # Apply transform to mask (same random decisions as image!)
        if self.transform_label is not None:
            mask = self.transform_label(mask)
            mask = mask.squeeze(0)
        else:
            mask = torch.from_numpy(mask).long()

        return image, mask

    def __len__(self):
        return len(self.image_paths)


def create_dataloaders(dataset_config, training_config, masks_dir):
    """
    Create train and validation dataloaders.

    Args:
        dataset_config: Dataset configuration dict
        training_config: Training configuration dict
        masks_dir: Path to training masks directory

    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
        train_images: List of training image paths (for prediction generation)
        base_transform: Base transform WITHOUT augmentation (for predictions)
        class_pixel_counts: Array of annotated pixel counts per class
    """
    print("Creating dataloaders...")

    # Check if dataset has normalization config
    normalization_config = dataset_config.get('normalization', None)

    if normalization_config:
        mean = normalization_config['mean']
        std = normalization_config['std']
        normalize = transforms.Normalize(mean=mean, std=std)
        print(f"  Normalization: ENABLED (dataset-specific)")
        print(f"    Mean: {mean}")
        print(f"    Std:  {std}")

        # Base transform (NO augmentation - for validation and prediction)
        base_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        # Training transform (WITH augmentation)
        # Perfect for aerial imagery where orientation doesn't matter
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            RandomRotation90()  # 0°, 90°, 180°, or 270° rotation
        ])
    else:
        print(f"  Normalization: DISABLED (no config found)")

        # Base transform (NO augmentation - for validation and prediction)
        base_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # Training transform (WITH augmentation)
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            RandomRotation90()  # 0°, 90°, 180°, or 270° rotation
        ])

    # Mask transform (same augmentation as image - synchronized by seed)
    # NOTE: Masks should NOT be normalized, only augmented
    train_transform_label = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        RandomRotation90()  # Same rotation as image (synchronized via random seed)
    ])

    print(f"  Augmentation: ENABLED (RandomHorizontalFlip, RandomVerticalFlip, RandomRotation90)")

    # Get train images and masks
    train_images_dir = Path(dataset_config['paths']['train_images'])
    train_images = sorted(list(train_images_dir.glob('*.png')) +
                         list(train_images_dir.glob('*.tif')) +
                         list(train_images_dir.glob('*.tiff')))
    train_masks = sorted(list(Path(masks_dir).glob('*.png')))

    print(f"  Train images: {len(train_images)}")
    print(f"  Train masks:  {len(train_masks)}")

    if len(train_images) != len(train_masks):
        raise ValueError(f"Mismatch: {len(train_images)} images but {len(train_masks)} masks")

    # Get validation images and masks
    val_images_dir = Path(dataset_config['paths']['val_images'])
    val_masks_dir = Path(dataset_config['paths']['val_masks'])
    val_images = sorted(list(val_images_dir.glob('*.png')) +
                       list(val_images_dir.glob('*.tif')) +
                       list(val_images_dir.glob('*.tiff')))
    val_masks = sorted(list(val_masks_dir.glob('*.png')))

    print(f"  Val images:   {len(val_images)}")
    print(f"  Val masks:    {len(val_masks)}")

    # Get ignore index from configuration
    ignore_index = dataset_config['classes']['ignore_index']
    print(f"  Ignore index: {ignore_index}")

    # Count annotated pixels per class
    num_classes = dataset_config['classes']['num_classes']
    class_names = dataset_config['classes']['names']
    class_pixel_counts = np.zeros(num_classes, dtype=np.int64)

    print(f"\n  Counting annotated pixels per class...")
    for mask_path in train_masks:
        mask = imageio.imread(mask_path)
        mask = np.asarray(mask, dtype='int64')
        for class_id in range(num_classes):
            class_pixel_counts[class_id] += np.sum(mask == class_id)

    print(f"  Annotated pixels by class:")
    total_pixels = 0
    for class_id in range(num_classes):
        count = class_pixel_counts[class_id]
        total_pixels += count
        if count > 0:
            print(f"    Class {class_id} ({class_names[class_id]}): {count:,} pixels")
        else:
            print(f"    Class {class_id} ({class_names[class_id]}): 0 pixels (no annotations)")
    print(f"  Total annotated pixels: {total_pixels:,}")
    print()

    # Create datasets
    train_dataset = CustomDataset(
        train_images,
        train_masks,
        ignore_index=ignore_index,
        transform=train_transform,         # Includes ToTensor + RandomFlips
        transform_label=train_transform_label  # Same augmentation (synchronized via seed!)
    )

    val_dataset = CustomDataset(
        val_images,
        val_masks,
        ignore_index=ignore_index,
        transform=base_transform,  # Just ToTensor, NO augmentation
        transform_label=None
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=training_config['training']['batch_size']['train'],
        shuffle=False,  # Like old notebook
        num_workers=0,
        pin_memory=True  # Faster CPU→GPU transfer
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=training_config['training']['batch_size']['val'],
        shuffle=False,
        num_workers=0,
        pin_memory=True  # Faster CPU→GPU transfer
    )

    print(f"✓ Dataloaders created")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}\n")

    return train_loader, val_loader, train_images, base_transform, class_pixel_counts
