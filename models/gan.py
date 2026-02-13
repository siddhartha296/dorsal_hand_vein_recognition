"""
Complete GAN Model
Combines Generator and Discriminator with training utilities
"""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict, Optional
import os

from models.generator import Generator
from models.discriminator import Discriminator
from config import config


class VeinGAN(nn.Module):
    """
    Complete GAN model for Dorsal Hand Vein Synthesis
    """
    
    def __init__(
        self,
        latent_dim: int = None,
        image_size: Tuple[int, int] = None,
        learning_rate_g: float = None,
        learning_rate_d: float = None,
        beta1: float = None,
        beta2: float = None
    ):
        """
        Args:
            latent_dim: Dimension of latent space
            image_size: Target image size
            learning_rate_g: Learning rate for generator
            learning_rate_d: Learning rate for discriminator
            beta1, beta2: Adam optimizer betas
        """
        super().__init__()
        
        # Load from config if not provided
        self.latent_dim = latent_dim or config.LATENT_DIM
        self.image_size = image_size or config.IMAGE_SIZE
        self.lr_g = learning_rate_g or config.LEARNING_RATE_G
        self.lr_d = learning_rate_d or config.LEARNING_RATE_D
        self.beta1 = beta1 or config.BETA1
        self.beta2 = beta2 or config.BETA2
        
        # Initialize networks
        self.generator = Generator(
            latent_dim=self.latent_dim,
            image_size=self.image_size
        )
        
        self.discriminator = Discriminator(
            image_size=self.image_size
        )
        
        # Move to device
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        
        # Initialize optimizers
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=self.lr_g,
            betas=(self.beta1, self.beta2)
        )
        
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr_d,
            betas=(self.beta1, self.beta2)
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        
        print(f"VeinGAN initialized on {self.device}")
        self._print_model_summary()
    
    def _print_model_summary(self):
        """Print model architecture summary"""
        g_params = sum(p.numel() for p in self.generator.parameters())
        d_params = sum(p.numel() for p in self.discriminator.parameters())
        
        print(f"\nModel Summary:")
        print(f"  Generator parameters: {g_params:,}")
        print(f"  Discriminator parameters: {d_params:,}")
        print(f"  Total parameters: {g_params + d_params:,}")
        print(f"  Latent dimension: {self.latent_dim}")
        print(f"  Image size: {self.image_size}")
    
    def generate(
        self,
        num_samples: int = 1,
        z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate synthetic vein images
        
        Args:
            num_samples: Number of images to generate
            z: Optional latent codes (if None, sample randomly)
        
        Returns:
            Generated images (B, C, H, W)
        """
        self.generator.eval()
        
        with torch.no_grad():
            if z is None:
                z = torch.randn(num_samples, self.latent_dim).to(self.device)
            
            fake_images = self.generator(z)
        
        return fake_images
    
    def discriminate(
        self,
        images: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Classify images as real or fake
        
        Args:
            images: Input images
            return_features: Whether to return features
        
        Returns:
            Classification logits (and features if requested)
        """
        self.discriminator.eval()
        
        with torch.no_grad():
            output = self.discriminator(images, return_features=return_features)
        
        return output
    
    def save_checkpoint(self, filepath: str, **kwargs):
        """
        Save model checkpoint
        
        Args:
            filepath: Path to save checkpoint
            **kwargs: Additional data to save
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'config': {
                'latent_dim': self.latent_dim,
                'image_size': self.image_size,
                'lr_g': self.lr_g,
                'lr_d': self.lr_d,
                'beta1': self.beta1,
                'beta2': self.beta2
            }
        }
        
        # Add any additional data
        checkpoint.update(kwargs)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str, load_optimizers: bool = True):
        """
        Load model checkpoint
        
        Args:
            filepath: Path to checkpoint
            load_optimizers: Whether to load optimizer states
        
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load model states
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        # Load optimizer states
        if load_optimizers:
            self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
            self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        
        print(f"Checkpoint loaded from {filepath}")
        print(f"  Epoch: {self.current_epoch}")
        print(f"  Global step: {self.global_step}")
        
        return checkpoint
    
    def get_latent_interpolation(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        steps: int = 10
    ) -> torch.Tensor:
        """
        Interpolate between two latent codes
        
        Args:
            z1: Starting latent code
            z2: Ending latent code
            steps: Number of interpolation steps
        
        Returns:
            Interpolated images
        """
        self.generator.eval()
        
        # Create interpolation weights
        alphas = torch.linspace(0, 1, steps).to(self.device)
        
        images = []
        with torch.no_grad():
            for alpha in alphas:
                z_interp = (1 - alpha) * z1 + alpha * z2
                img = self.generator(z_interp)
                images.append(img)
        
        return torch.cat(images, dim=0)
    
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requires_grad for network parameters"""
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad


class FeatureExtractor(nn.Module):
    """
    Feature extractor for vein authentication
    Uses trained discriminator as feature extractor
    """
    
    def __init__(self, discriminator: Discriminator):
        super().__init__()
        self.discriminator = discriminator
        self.discriminator.eval()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from vein image"""
        with torch.no_grad():
            features = self.discriminator.extract_features(x)
        return features
    
    def compute_similarity(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        metric: str = 'cosine'
    ) -> torch.Tensor:
        """
        Compute similarity between two vein images
        
        Args:
            img1, img2: Input images
            metric: Similarity metric ('cosine', 'euclidean', 'manhattan')
        
        Returns:
            Similarity score
        """
        feat1 = self.forward(img1)
        feat2 = self.forward(img2)
        
        if metric == 'cosine':
            # Cosine similarity
            similarity = nn.functional.cosine_similarity(feat1, feat2, dim=1)
        elif metric == 'euclidean':
            # Negative Euclidean distance (higher = more similar)
            similarity = -torch.norm(feat1 - feat2, p=2, dim=1)
        elif metric == 'manhattan':
            # Negative Manhattan distance
            similarity = -torch.norm(feat1 - feat2, p=1, dim=1)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return similarity


def test_vein_gan():
    """Test VeinGAN"""
    print("Testing VeinGAN...")
    
    # Create model
    gan = VeinGAN()
    
    # Test generation
    num_samples = 4
    fake_images = gan.generate(num_samples)
    print(f"\nGenerated images shape: {fake_images.shape}")
    print(f"Value range: [{fake_images.min():.3f}, {fake_images.max():.3f}]")
    
    # Test discrimination
    logits = gan.discriminate(fake_images)
    print(f"Discriminator logits shape: {logits.shape}")
    
    # Test checkpoint saving/loading
    checkpoint_path = config.CHECKPOINT_DIR / "test_checkpoint.pth"
    gan.save_checkpoint(checkpoint_path, test_metric=0.95)
    
    # Create new model and load
    gan2 = VeinGAN()
    gan2.load_checkpoint(checkpoint_path)
    
    print("\nTest completed successfully!")
    
    return gan


if __name__ == "__main__":
    test_vein_gan()