"""
Data Augmentation for Dorsal Hand Vein Images
"""
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
import random
from typing import Tuple

from config import config


class VeinImageAugmentation:
    """
    Augmentation pipeline for vein images
    Preserves vein patterns while adding realistic variations
    """
    
    def __init__(
        self,
        rotation_range: float = 15,
        width_shift_range: float = 0.1,
        height_shift_range: float = 0.1,
        horizontal_flip: bool = True,
        zoom_range: float = 0.1,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        add_noise: bool = True,
        noise_std: float = 0.05
    ):
        """
        Args:
            rotation_range: Degrees of rotation
            width_shift_range: Fraction of width to shift
            height_shift_range: Fraction of height to shift
            horizontal_flip: Whether to apply horizontal flipping
            zoom_range: Fraction to zoom in/out
            brightness_range: Range for brightness adjustment
            add_noise: Whether to add Gaussian noise
            noise_std: Standard deviation of noise
        """
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.horizontal_flip = horizontal_flip
        self.zoom_range = zoom_range
        self.brightness_range = brightness_range
        self.add_noise = add_noise
        self.noise_std = noise_std
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to image
        
        Args:
            image: Input image tensor (C, H, W)
        
        Returns:
            Augmented image tensor
        """
        # Random rotation
        if self.rotation_range > 0:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            image = TF.rotate(image, angle)
        
        # Random horizontal flip
        if self.horizontal_flip and random.random() < 0.5:
            image = TF.hflip(image)
        
        # Random affine transformation (shift and zoom)
        if self.width_shift_range > 0 or self.height_shift_range > 0 or self.zoom_range > 0:
            # Calculate parameters
            h, w = image.shape[-2:]
            
            # Translation
            max_dx = int(w * self.width_shift_range)
            max_dy = int(h * self.height_shift_range)
            dx = random.randint(-max_dx, max_dx)
            dy = random.randint(-max_dy, max_dy)
            
            # Zoom (scale)
            if self.zoom_range > 0:
                scale = random.uniform(1 - self.zoom_range, 1 + self.zoom_range)
            else:
                scale = 1.0
            
            # Apply affine transformation
            image = TF.affine(
                image,
                angle=0,
                translate=(dx, dy),
                scale=scale,
                shear=0
            )
        
        # Brightness adjustment
        if self.brightness_range:
            brightness_factor = random.uniform(*self.brightness_range)
            image = TF.adjust_brightness(image, brightness_factor)
        
        # Add Gaussian noise
        if self.add_noise:
            noise = torch.randn_like(image) * self.noise_std
            image = image + noise
            image = torch.clamp(image, -1, 1)  # Assuming normalized to [-1, 1]
        
        return image


class ElasticTransform:
    """
    Elastic deformation for vein images
    Creates realistic distortions while preserving vein structure
    """
    
    def __init__(self, alpha: float = 10, sigma: float = 3):
        """
        Args:
            alpha: Scaling factor for deformation
            sigma: Gaussian filter sigma
        """
        self.alpha = alpha
        self.sigma = sigma
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply elastic transformation"""
        # Convert to numpy for scipy operations
        import scipy.ndimage as ndi
        
        image_np = image.squeeze().cpu().numpy()
        shape = image_np.shape
        
        # Generate random displacement fields
        dx = ndi.gaussian_filter(
            (np.random.rand(*shape) * 2 - 1),
            self.sigma,
            mode="constant",
            cval=0
        ) * self.alpha
        
        dy = ndi.gaussian_filter(
            (np.random.rand(*shape) * 2 - 1),
            self.sigma,
            mode="constant",
            cval=0
        ) * self.alpha
        
        # Create meshgrid
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = (y + dy).reshape(-1), (x + dx).reshape(-1)
        
        # Apply deformation
        distorted = ndi.map_coordinates(
            image_np,
            indices,
            order=1,
            mode='reflect'
        ).reshape(shape)
        
        # Convert back to tensor
        distorted_tensor = torch.from_numpy(distorted).unsqueeze(0).float()
        
        return distorted_tensor


class VeinSpecificAugmentation:
    """
    Augmentation specifically designed for vein patterns
    - Simulates different illumination conditions
    - Simulates hand positioning variations
    """
    
    def __init__(self):
        pass
    
    def add_shadow(self, image: torch.Tensor, intensity: float = 0.3) -> torch.Tensor:
        """Add realistic shadow to simulate different lighting"""
        h, w = image.shape[-2:]
        
        # Create random shadow mask
        shadow_mask = torch.zeros((h, w))
        
        # Random elliptical shadow
        center_x = random.randint(w // 4, 3 * w // 4)
        center_y = random.randint(h // 4, 3 * h // 4)
        radius_x = random.randint(w // 4, w // 2)
        radius_y = random.randint(h // 4, h // 2)
        
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        ellipse = ((x - center_x) / radius_x) ** 2 + ((y - center_y) / radius_y) ** 2
        shadow_mask = torch.exp(-ellipse * 2)
        
        # Apply shadow
        shadow_mask = shadow_mask.unsqueeze(0) * intensity
        image = image * (1 - shadow_mask)
        
        return image
    
    def simulate_pressure_variation(
        self,
        image: torch.Tensor,
        intensity: float = 0.2
    ) -> torch.Tensor:
        """Simulate different hand pressure on sensor"""
        # Adjust contrast to simulate pressure
        contrast_factor = 1 + random.uniform(-intensity, intensity)
        mean = image.mean()
        image = (image - mean) * contrast_factor + mean
        image = torch.clamp(image, -1, 1)
        
        return image


def get_train_augmentation() -> VeinImageAugmentation:
    """Get augmentation pipeline for training"""
    if config.USE_AUGMENTATION:
        return VeinImageAugmentation(
            rotation_range=config.AUGMENTATION_PARAMS['rotation_range'],
            width_shift_range=config.AUGMENTATION_PARAMS['width_shift_range'],
            height_shift_range=config.AUGMENTATION_PARAMS['height_shift_range'],
            horizontal_flip=config.AUGMENTATION_PARAMS['horizontal_flip'],
            zoom_range=config.AUGMENTATION_PARAMS['zoom_range'],
            brightness_range=config.AUGMENTATION_PARAMS['brightness_range']
        )
    else:
        return None


def get_test_augmentation() -> None:
    """No augmentation for test/validation"""
    return None


# Additional utility transforms
def normalize_vein_image(image: torch.Tensor) -> torch.Tensor:
    """Normalize vein image to [-1, 1] range"""
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    image = image * 2.0 - 1.0
    return image


def denormalize_vein_image(image: torch.Tensor) -> torch.Tensor:
    """Denormalize from [-1, 1] to [0, 1]"""
    image = (image + 1.0) / 2.0
    return torch.clamp(image, 0, 1)


if __name__ == "__main__":
    # Test augmentation
    print("Testing Vein Image Augmentation...")
    
    # Create dummy image
    dummy_image = torch.randn(1, 256, 256)
    
    # Test augmentation
    aug = get_train_augmentation()
    if aug:
        augmented = aug(dummy_image)
        print(f"Original shape: {dummy_image.shape}")
        print(f"Augmented shape: {augmented.shape}")
        print(f"Value range: [{augmented.min():.3f}, {augmented.max():.3f}]")
    
    # Test vein-specific augmentation
    vein_aug = VeinSpecificAugmentation()
    shadowed = vein_aug.add_shadow(dummy_image)
    print(f"Shadow applied: {shadowed.shape}")