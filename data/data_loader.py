"""
Data loader for Dorsal Hand Vein Images
Handles loading, preprocessing, and batching of the vein image dataset
"""
import os
import re
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split

from config import config


class DorsalHandVeinDataset(Dataset):
    """
    PyTorch Dataset for Dorsal Hand Vein Images
    
    Supports both Database 1 and Database 2
    Naming convention: person_[xxx]_db[1|2]_[L|R][1-4].tif
    """
    
    def __init__(
        self,
        db_paths: List[Path],
        transform=None,
        normalize=True,
        image_size: Tuple[int, int] = None
    ):
        """
        Args:
            db_paths: List of paths to database directories
            transform: Optional transform to be applied on images
            normalize: Whether to normalize images to [-1, 1]
            image_size: Target image size (height, width)
        """
        self.db_paths = db_paths if isinstance(db_paths, list) else [db_paths]
        self.transform = transform
        self.normalize = normalize
        self.image_size = image_size or config.IMAGE_SIZE
        
        # Collect all image paths and metadata
        self.image_data = self._collect_image_data()
        
        print(f"Loaded {len(self.image_data)} images from {len(self.db_paths)} database(s)")
        self._print_dataset_statistics()
    
    def _collect_image_data(self) -> List[Dict]:
        """Collect all image paths and parse metadata"""
        image_data = []
        
        for db_path in self.db_paths:
            if not db_path.exists():
                print(f"Warning: Database path does not exist: {db_path}")
                continue
            
            # Find all .tif files
            tif_files = sorted(db_path.glob("*.tif"))
            
            for img_path in tif_files:
                # Parse filename: person_[xxx]_db[1|2]_[L|R][1-4].tif
                filename = img_path.stem
                match = re.match(r'person_(\d+)_db(\d)_([LR])(\d)', filename)
                
                if match:
                    person_id, db_num, hand_side, image_num = match.groups()
                    
                    image_data.append({
                        'path': img_path,
                        'person_id': int(person_id),
                        'database': int(db_num),
                        'hand_side': hand_side,  # 'L' or 'R'
                        'image_num': int(image_num),
                        'filename': filename
                    })
        
        return image_data
    
    def _print_dataset_statistics(self):
        """Print dataset statistics"""
        if not self.image_data:
            return
        
        person_ids = set(d['person_id'] for d in self.image_data)
        left_hands = sum(1 for d in self.image_data if d['hand_side'] == 'L')
        right_hands = sum(1 for d in self.image_data if d['hand_side'] == 'R')
        db1_count = sum(1 for d in self.image_data if d['database'] == 1)
        db2_count = sum(1 for d in self.image_data if d['database'] == 2)
        
        print(f"\nDataset Statistics:")
        print(f"  Total images: {len(self.image_data)}")
        print(f"  Unique persons: {len(person_ids)}")
        print(f"  Left hands: {left_hands}")
        print(f"  Right hands: {right_hands}")
        print(f"  Database 1: {db1_count}")
        print(f"  Database 2: {db2_count}")
    
    def __len__(self) -> int:
        return len(self.image_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Returns:
            image: Preprocessed image tensor
            metadata: Dictionary with image metadata
        """
        data = self.image_data[idx]
        
        # Load image
        image = self._load_image(data['path'])
        
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
        
        # Convert to tensor and normalize
        image = self._preprocess_image(image)
        
        # Prepare metadata
        metadata = {
            'person_id': data['person_id'],
            'hand_side': data['hand_side'],
            'database': data['database'],
            'filename': data['filename']
        }
        
        return image, metadata
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Load 16-bit TIFF image"""
        # Using PIL for TIFF loading
        img = Image.open(path)
        img_array = np.array(img, dtype=np.float32)
        
        # Resize if necessary
        if img_array.shape[:2] != self.image_size:
            img_array = cv2.resize(
                img_array,
                (self.image_size[1], self.image_size[0]),
                interpolation=cv2.INTER_AREA
            )
        
        return img_array
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image to tensor"""
        # Normalize to [0, 1]
        image = image / image.max() if image.max() > 0 else image
        
        # Normalize to [-1, 1] if specified (for GAN training)
        if self.normalize:
            image = image * 2.0 - 1.0
        
        # Add channel dimension and convert to tensor
        image = torch.from_numpy(image).unsqueeze(0).float()
        
        return image
    
    def get_person_images(self, person_id: int) -> List[int]:
        """Get all image indices for a specific person"""
        return [i for i, d in enumerate(self.image_data) 
                if d['person_id'] == person_id]
    
    def get_hand_images(self, hand_side: str) -> List[int]:
        """Get all image indices for a specific hand side ('L' or 'R')"""
        return [i for i, d in enumerate(self.image_data) 
                if d['hand_side'] == hand_side]


def create_data_loaders(
    db_paths: List[Path],
    batch_size: int = None,
    train_split: float = None,
    val_split: float = None,
    test_split: float = None,
    random_seed: int = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders
    
    Args:
        db_paths: List of database paths
        batch_size: Batch size for data loaders
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        random_seed: Random seed for reproducibility
    
    Returns:
        train_loader, val_loader, test_loader
    """
    batch_size = batch_size or config.BATCH_SIZE
    train_split = train_split or config.TRAIN_SPLIT
    val_split = val_split or config.VAL_SPLIT
    test_split = test_split or config.TEST_SPLIT
    random_seed = random_seed or config.RANDOM_SEED
    
    # Create full dataset
    full_dataset = DorsalHandVeinDataset(db_paths)
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=generator
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True  # Drop last incomplete batch for GAN training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    print(f"\nData Loaders Created:")
    print(f"  Train: {len(train_dataset)} samples ({len(train_loader)} batches)")
    print(f"  Val: {len(val_dataset)} samples ({len(val_loader)} batches)")
    print(f"  Test: {len(test_dataset)} samples ({len(test_loader)} batches)")
    
    return train_loader, val_loader, test_loader


def get_paired_verification_loader(
    db_paths: List[Path],
    batch_size: int = None,
    num_pairs: int = 1000
) -> DataLoader:
    """
    Create a data loader for verification tasks with positive/negative pairs
    
    Args:
        db_paths: List of database paths
        batch_size: Batch size
        num_pairs: Number of pairs to generate
    
    Returns:
        DataLoader with pairs
    """
    from torch.utils.data import TensorDataset
    
    batch_size = batch_size or config.BATCH_SIZE
    
    # Load dataset
    dataset = DorsalHandVeinDataset(db_paths)
    
    # Generate pairs
    pairs_1, pairs_2, labels = [], [], []
    
    person_ids = list(set(d['person_id'] for d in dataset.image_data))
    
    for _ in range(num_pairs):
        # Positive pair (same person)
        if np.random.rand() < 0.5:
            person_id = np.random.choice(person_ids)
            person_images = dataset.get_person_images(person_id)
            
            if len(person_images) >= 2:
                idx1, idx2 = np.random.choice(person_images, 2, replace=False)
                img1, _ = dataset[idx1]
                img2, _ = dataset[idx2]
                
                pairs_1.append(img1)
                pairs_2.append(img2)
                labels.append(1)  # Same person
        
        # Negative pair (different persons)
        else:
            person1, person2 = np.random.choice(person_ids, 2, replace=False)
            images1 = dataset.get_person_images(person1)
            images2 = dataset.get_person_images(person2)
            
            idx1 = np.random.choice(images1)
            idx2 = np.random.choice(images2)
            
            img1, _ = dataset[idx1]
            img2, _ = dataset[idx2]
            
            pairs_1.append(img1)
            pairs_2.append(img2)
            labels.append(0)  # Different persons
    
    # Create tensor dataset
    pairs_1 = torch.stack(pairs_1)
    pairs_2 = torch.stack(pairs_2)
    labels = torch.tensor(labels, dtype=torch.float32)
    
    pair_dataset = TensorDataset(pairs_1, pairs_2, labels)
    
    pair_loader = DataLoader(
        pair_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )
    
    return pair_loader


if __name__ == "__main__":
    # Test the data loader
    print("Testing Dorsal Hand Vein Data Loader...")
    
    # Example usage
    db_paths = [config.DB1_PATH, config.DB2_PATH]
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(db_paths)
    
    # Test loading a batch
    batch, metadata = next(iter(train_loader))
    print(f"\nBatch shape: {batch.shape}")
    print(f"Batch min/max: {batch.min():.3f} / {batch.max():.3f}")
    print(f"Metadata keys: {metadata.keys()}")