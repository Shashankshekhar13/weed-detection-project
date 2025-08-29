import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch

def load_datasets(data_dir, batch_size=32):
    """
    Loads dataset, applies transforms, splits into train/val/test,
    and returns (train_loader, val_loader, test_loader, class_names).
    """
    # Define transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Load and split
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)
    class_names = full_dataset.classes
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Apply test transforms to val/test
    val_dataset.dataset.transform = test_transforms
    test_dataset.dataset.transform = test_transforms

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2)

    # Print summary
    print(f"✅ Dataset Loaded from: {data_dir}")
    print(f"Classes: {class_names}")
    print(f"Total Images: {total_size}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader, class_names
