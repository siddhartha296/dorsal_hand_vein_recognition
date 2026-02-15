"""
FIXED Data loader for Dorsal Hand Vein Images
This version correctly handles tensor dimensions
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

from config import config


class DorsalHandVeinDataset(Dataset):
    """
    PyTorch Dataset for Dorsal Hand Vein Images - FIXED VERSION
    
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
            image: Preprocessed image tensor of shape (1, H, W)
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
        
        # CRITICAL: Verify shape before returning
        assert image.ndim == 3, f"Expected 3D tensor, got {image.ndim}D: {image.shape}"
        assert image.shape[0] == 1, f"Expected 1 channel, got {image.shape[0]}: {image.shape}"
        
        # Prepare metadata
        metadata = {
            'person_id': data['person_id'],
            'hand_side': data['hand_side'],
            'database': data['database'],
            'filename': data['filename']
        }
        
        return image, metadata
    
    def _load_image(self, path: Path) -> np.ndarray:
        """
        Load 16-bit TIFF image
        Returns: numpy array of shape (H, W) - 2D grayscale
        """
        # Using PIL for TIFF loading
        img = Image.open(path)
        
        # Convert to grayscale if not already
        if img.mode != 'L':
            img = img.convert('L')
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)
        
        # CRITICAL: Ensure 2D array
        if img_array.ndim == 3:
            # If 3D, take first channel
            img_array = img_array[:, :, 0]
        
        # Resize if necessary
        if img_array.shape[:2] != self.image_size:
            img_array = cv2.resize(
                img_array,
                (self.image_size[1], self.image_size[0]),
                interpolation=cv2.INTER_AREA
            )
        
        return img_array
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image to tensor - FIXED VERSION
        
        Input: numpy array of shape (H, W) - grayscale image
        Output: tensor of shape (1, H, W) - single channel
        """
        # CRITICAL: Ensure image is 2D (H, W)
        if image.ndim == 3:
            # If accidentally loaded as RGB, convert to grayscale
            if image.shape[2] == 3:
                # Take mean across channels
                image = image.mean(axis=2)
            elif image.shape[2] == 1:
                # Remove extra dimension
                image = image.squeeze(2)
            else:
                # Take first channel
                image = image[:, :, 0]
        
        # Ensure still 2D after operations
        while image.ndim > 2:
            image = image.squeeze()
        
        if image.ndim < 2:
            raise ValueError(f"Image has invalid dimensions: {image.shape}")
        
        # Normalize to [0, 1]
        if image.max() > 0:
            image = image / image.max()
        
        # Normalize to [-1, 1] if specified (for GAN training)
        if self.normalize:
            image = image * 2.0 - 1.0
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).float()
        
        # CRITICAL: Add channel dimension to get (1, H, W)
        if image_tensor.ndim == 2:
            image_tensor = image_tensor.unsqueeze(0)  # (H, W) -> (1, H, W)
        
        # Final verification
        if image_tensor.ndim != 3 or image_tensor.shape[0] != 1:
            raise ValueError(
                f"Tensor has wrong shape: {image_tensor.shape}. "
                f"Expected (1, H, W)"
            )
        
        return image_tensor
    
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
    
    # CRITICAL: Verify data shapes
    _verify_dataloader_output(train_loader)
    
    return train_loader, val_loader, test_loader


def _verify_dataloader_output(data_loader: DataLoader):
    """Verify that dataloader produces correctly shaped tensors"""
    print("\n" + "="*60)
    print("VERIFYING DATALOADER OUTPUT")
    print("="*60)
    
    batch, metadata = next(iter(data_loader))
    
    print(f"Batch shape: {batch.shape}")
    print(f"Expected: (batch_size, 1, H, W)")
    print(f"Value range: [{batch.min():.3f}, {batch.max():.3f}]")
    
    # Critical assertions
    assert batch.ndim == 4, f"Expected 4D batch tensor, got {batch.ndim}D: {batch.shape}"
    assert batch.shape[1] == 1, f"Expected 1 channel, got {batch.shape[1]}"
    assert batch.shape[2:] == config.IMAGE_SIZE, f"Wrong spatial dimensions: {batch.shape[2:]}"
    
    print("✓ Dataloader output verification PASSED")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Test the data loader
    print("Testing FIXED Dorsal Hand Vein Data Loader...")
    
    # Example usage
    db_paths = [config.DB1_PATH, config.DB2_PATH]
    
    try:
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(db_paths)
        
        # Test loading a batch
        batch, metadata = next(iter(train_loader))
        print(f"\n✓ SUCCESS!")
        print(f"Batch shape: {batch.shape}")
        print(f"Batch min/max: {batch.min():.3f} / {batch.max():.3f}")
        print(f"Metadata keys: {metadata.keys()}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()