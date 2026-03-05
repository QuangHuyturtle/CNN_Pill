"""
Dataset module for ePillID - Pill Image Classification
Supports 5-fold cross-validation with proper data augmentation
"""

import os
import pickle
from typing import Dict, List, Tuple, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class PillDataset(Dataset):
    """
    Dataset for pill image classification.

    Args:
        data_df: DataFrame containing image information with columns:
                 - image_path: relative path to image
                 - label: class label (pill type)
        image_dir: Root directory containing images
        transform: Optional transform to apply to images
        label_encoder: Dictionary mapping label strings to integers
    """

    def __init__(
        self,
        data_df: pd.DataFrame,
        image_dir: str,
        transform: Optional[transforms.Compose] = None,
        label_encoder: Optional[Dict[str, int]] = None
    ):
        self.image_dir = image_dir
        self.transform = transform

        # Filter out rows where image file doesn't exist
        valid_rows = []
        for _, row in data_df.iterrows():
            filename = row['images']
            img_path = os.path.join(image_dir, filename)
            if os.path.exists(img_path):
                valid_rows.append(row)

        self.data_df = pd.DataFrame(valid_rows).reset_index(drop=True)

        if len(self.data_df) < len(data_df):
            print(f"Warning: Filtered out {len(data_df) - len(self.data_df)} images with missing files")

        # Build label encoder if not provided
        if label_encoder is None:
            unique_labels = sorted(self.data_df['label'].unique())
            self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_encoder = label_encoder

        self.num_classes = len(self.label_encoder)

    def __len__(self) -> int:
        return len(self.data_df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.data_df.iloc[idx]

        # Get image filename from 'images' column (the actual filename)
        # CSV structure: 'images' column has filename like '0.jpg', '100.jpg'
        filename = row['images']

        # Load image
        img_path = os.path.join(self.image_dir, filename)
        image = self._load_image(img_path)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Get label
        label_str = row['label']
        label = self.label_encoder[label_str]

        return image, label

    def _load_image(self, img_path: str) -> Image.Image:
        """Load image from path, handling errors gracefully."""
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Load with PIL (works better with torchvision transforms)
        image = Image.open(img_path).convert('RGB')
        return image


class DataManager:
    """
    Manages data loading and 5-fold cross-validation splits.

    Args:
        data_dir: Directory containing the dataset
        fold_dir: Directory containing fold CSV files
        fold_name: Name of the fold configuration
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for data loading
        image_size: Size to resize images to (default 224 for EfficientNet)
    """

    def __init__(
        self,
        data_dir: str = "data",
        fold_dir: str = None,
        fold_name: str = "pilltypeid_nih_sidelbls0.01_metric_5folds",
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224
    ):
        self.data_dir = data_dir
        self.fold_name = fold_name
        # Build fold_dir from data_dir if not provided
        if fold_dir is None:
            self.fold_dir = os.path.join(data_dir, "folds", fold_name, "base")
        else:
            self.fold_dir = fold_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        # Load label encoder (build from CSV if pickle fails)
        self.label_encoder = self._load_or_build_encoder()

        # Inverse encoder for inference
        self.idx_to_label = {v: k for k, v in self.label_encoder.items()}
        self.num_classes = len(self.label_encoder)

        print(f"Loaded label encoder with {self.num_classes} classes")

        # Load fold data
        self.folds = self._load_folds()

    def _load_or_build_encoder(self) -> Dict[str, int]:
        """Load label encoder from pickle or build from fold CSV files."""
        encoder_path = os.path.join(self.fold_dir, "label_encoder.pickle")

        # Try loading from pickle first
        if os.path.exists(encoder_path):
            try:
                with open(encoder_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Warning: Could not load pickle ({e}), building from CSV...")

        # Build from fold CSV files
        print("Building label encoder from fold CSV files...")
        all_labels = set()

        # Load all folds to get unique labels
        for i in range(5):
            fold_path = os.path.join(self.fold_dir, f"{self.fold_name}_{i}.csv")
            if not os.path.exists(fold_path):
                continue
            df = pd.read_csv(fold_path)
            if 'label' in df.columns:
                all_labels.update(df['label'].unique())

        # Create encoder
        sorted_labels = sorted(all_labels)
        encoder = {label: idx for idx, label in enumerate(sorted_labels)}

        # Save for future use
        try:
            with open(encoder_path, 'wb') as f:
                pickle.dump(encoder, f)
            print(f"Saved label encoder to {encoder_path}")
        except Exception as e:
            print(f"Warning: Could not save encoder ({e})")

        return encoder

    def _load_folds(self) -> List[pd.DataFrame]:
        """Load all 5 fold CSV files."""
        folds = []
        for i in range(5):
            fold_path = os.path.join(self.fold_dir, f"{self.fold_name}_{i}.csv")
            if os.path.exists(fold_path):
                folds.append(pd.read_csv(fold_path))
            else:
                print(f"Warning: Fold file not found: {fold_path}")
        return folds

    def get_train_val_data(self, fold_idx: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get train and validation data for a specific fold.

        Args:
            fold_idx: Which fold to use as validation (0-4)

        Returns:
            train_df, val_df: DataFrames for training and validation
        """
        val_df = self.folds[fold_idx]

        # Concatenate all other folds for training
        train_dfs = [self.folds[i] for i in range(5) if i != fold_idx]
        train_df = pd.concat(train_dfs, ignore_index=True)

        print(f"Fold {fold_idx}: Train={len(train_df)}, Val={len(val_df)}")
        return train_df, val_df

    def get_data_loaders(
        self,
        fold_idx: int = 0,
        augmentation: bool = True
    ) -> Tuple[DataLoader, DataLoader, int]:
        """
        Get train and validation DataLoaders for a specific fold.

        Args:
            fold_idx: Which fold to use as validation (0-4)
            augmentation: Whether to apply data augmentation to training set

        Returns:
            train_loader, val_loader: PyTorch DataLoaders, num_classes
        """
        train_df, val_df = self.get_train_val_data(fold_idx)

        # Find common labels between train and val
        train_labels = set(train_df['label'].unique())
        val_labels = set(val_df['label'].unique())
        common_labels = train_labels.intersection(val_labels)

        if len(common_labels) == 0:
            raise ValueError(f"Fold {fold_idx}: No common labels between train and val! Train has {len(train_labels)} classes, Val has {len(val_labels)} classes.")

        # Filter both train and val to only use common labels
        train_df = train_df[train_df['label'].isin(common_labels)].reset_index(drop=True)
        val_df = val_df[val_df['label'].isin(common_labels)].reset_index(drop=True)

        # Build label encoder from common labels
        unique_labels = sorted(common_labels)
        fold_label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        num_classes = len(fold_label_encoder)

        print(f"Fold {fold_idx}: Using {num_classes} common labels (from {len(train_labels)} train classes, {len(val_labels)} val classes)")
        print(f"Fold {fold_idx}: Train={len(train_df)}, Val={len(val_df)} samples")

        # Training transforms with aggressive augmentation to reduce overfitting
        train_transform = transforms.Compose([
            transforms.Resize((int(self.image_size * 1.1), int(self.image_size * 1.1))),  # Slightly larger for random crop
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=30),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.15, 0.15),
                scale=(0.85, 1.15),
                shear=5
            ),
            transforms.RandomResizedCrop(self.image_size, scale=(0.7, 1.0)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        # Validation transforms (no augmentation)
        val_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        if not augmentation:
            train_transform = val_transform

        # Update image path - use fcn_mix_weight/dc_224 where the actual images are
        image_base_dir = os.path.join(self.data_dir, "ePillID_data/classification_data/fcn_mix_weight/dc_224")

        # Create datasets with fold-specific label encoder
        train_dataset = PillDataset(
            data_df=train_df,
            image_dir=image_base_dir,
            transform=train_transform,
            label_encoder=fold_label_encoder
        )

        val_dataset = PillDataset(
            data_df=val_df,
            image_dir=image_base_dir,
            transform=val_transform,
            label_encoder=fold_label_encoder
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        return train_loader, val_loader, num_classes

    def get_class_weights(self, fold_idx: int = 0, num_classes: int = None) -> torch.Tensor:
        """
        Calculate class weights for imbalanced dataset.
        Useful for loss function weighting.

        Args:
            fold_idx: Which fold to calculate weights for
            num_classes: Number of classes in this fold (must match fold-specific encoder)
        """
        train_df, val_df = self.get_train_val_data(fold_idx)

        # Find common labels between train and val (same logic as get_data_loaders)
        train_labels = set(train_df['label'].unique())
        val_labels = set(val_df['label'].unique())
        common_labels = train_labels.intersection(val_labels)

        # Filter train_df to only use common labels
        train_df = train_df[train_df['label'].isin(common_labels)].reset_index(drop=True)

        # Build fold-specific label encoder from common labels
        unique_labels = sorted(common_labels)
        fold_label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        fold_num_classes = len(fold_label_encoder)

        # Use provided num_classes if available, otherwise use fold-specific count
        if num_classes is not None:
            fold_num_classes = num_classes

        # Map labels to fold-specific indices and count
        label_counts = train_df['label'].map(fold_label_encoder).value_counts()

        # Calculate weights (inverse frequency)
        total_samples = len(train_df)
        class_weights = []

        for idx in range(fold_num_classes):
            count = label_counts.get(idx, 1)  # Avoid division by zero
            weight = total_samples / (fold_num_classes * count)
            class_weights.append(weight)

        return torch.FloatTensor(class_weights)

    def get_all_data_loaders(
        self,
        augmentation: bool = True,
        train_ratio: float = 0.8,
        random_seed: int = 42
    ) -> Tuple[DataLoader, DataLoader, int]:
        """
        Load all folds and do random train/val split.
        This is the recommended approach for ePillID dataset since folds don't share labels.

        Args:
            augmentation: Whether to apply data augmentation to training set
            train_ratio: Ratio of training data (default 0.8 = 80%)
            random_seed: Random seed for reproducibility

        Returns:
            train_loader, val_loader: PyTorch DataLoaders, num_classes
        """
        from sklearn.model_selection import train_test_split

        # Combine all folds
        all_folds_df = pd.concat(self.folds, ignore_index=True)
        print(f"Loaded all folds: {len(all_folds_df)} total samples")

        # Get unique labels and build encoder
        unique_labels = sorted(all_folds_df['label'].unique())
        label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        num_classes = len(label_encoder)
        print(f"Total unique classes: {num_classes}")

        # Split into train and val (simple random split without stratify)
        # Note: Can't use stratify because some classes have only 1 sample
        train_df, val_df = train_test_split(
            all_folds_df,
            train_size=train_ratio,
            random_state=random_seed
        )

        print(f"Random split: Train={len(train_df)}, Val={len(val_df)}")

        # Training transforms with aggressive augmentation to reduce overfitting
        train_transform = transforms.Compose([
            transforms.Resize((int(self.image_size * 1.1), int(self.image_size * 1.1))),  # Slightly larger for random crop
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=30),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.15, 0.15),
                scale=(0.85, 1.15),
                shear=5
            ),
            transforms.RandomResizedCrop(self.image_size, scale=(0.7, 1.0)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        # Validation transforms (no augmentation)
        val_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        if not augmentation:
            train_transform = val_transform

        # Image directory
        image_base_dir = os.path.join(self.data_dir, "ePillID_data/classification_data/fcn_mix_weight/dc_224")

        # Create datasets
        train_dataset = PillDataset(
            data_df=train_df,
            image_dir=image_base_dir,
            transform=train_transform,
            label_encoder=label_encoder
        )

        val_dataset = PillDataset(
            data_df=val_df,
            image_dir=image_base_dir,
            transform=val_transform,
            label_encoder=label_encoder
        )

        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Val dataset size: {len(val_dataset)}")

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        return train_loader, val_loader, num_classes

    def get_all_class_weights(self, num_classes: int = None) -> torch.Tensor:
        """
        Calculate class weights for the entire dataset.
        Use with get_all_data_loaders().

        Args:
            num_classes: Number of classes (should match get_all_data_loaders result)
        """
        # Combine all folds
        all_folds_df = pd.concat(self.folds, ignore_index=True)

        # Build label encoder (same as get_all_data_loaders)
        unique_labels = sorted(all_folds_df['label'].unique())
        label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        fold_num_classes = len(label_encoder)

        if num_classes is not None:
            fold_num_classes = num_classes

        # Encode and count
        all_folds_df['encoded_label'] = all_folds_df['label'].map(label_encoder)
        label_counts = all_folds_df['encoded_label'].value_counts()

        # Calculate weights (inverse frequency)
        total_samples = len(all_folds_df)
        class_weights = []

        for idx in range(fold_num_classes):
            count = label_counts.get(idx, 1)
            weight = total_samples / (fold_num_classes * count)
            class_weights.append(weight)

        return torch.FloatTensor(class_weights)


def get_transforms(image_size: int = 224, train: bool = True) -> transforms.Compose:
    """
    Get image transforms for training or inference.

    Args:
        image_size: Target image size
        train: Whether this is for training (applies augmentation)

    Returns:
        Transforms composition
    """
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


if __name__ == "__main__":
    # Test the data module
    print("Testing DataManager...")

    data_manager = DataManager(
        data_dir="../ePillID_data",
        batch_size=16
    )

    # Test fold 0
    train_loader, val_loader = data_manager.get_data_loaders(fold_idx=0)

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Test loading a batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label range: {labels.min()} - {labels.max()}")
