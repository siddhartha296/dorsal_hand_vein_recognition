"""
Discriminator Network with Spectral Normalization
For XAI-Enhanced GAN for Dorsal Hand Vein Authentication
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

from config import config


class SpectralNorm(nn.Module):
    """
    Spectral Normalization wrapper
    Stabilizes GAN training by constraining weight matrices
    """
    
    def __init__(self, module: nn.Module, name: str = 'weight', power_iterations: int = 1):
        super().__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        
        if not self._made_params():
            self._make_params()
    
    def _made_params(self):
        try:
            getattr(self.module, self.name + "_u")
            getattr(self.module, self.name + "_v")
            getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False
    
    def _make_params(self):
        w = getattr(self.module, self.name)
        
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        
        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = self._l2normalize(u.data)
        v.data = self._l2normalize(v.data)
        w_bar = nn.Parameter(w.data)
        
        del self.module._parameters[self.name]
        
        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)
    
    def _l2normalize(self, v, eps=1e-12):
        return v / (v.norm() + eps)
    
    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")
        
        height = w.data.shape[0]
        
        for _ in range(self.power_iterations):
            v.data = self._l2normalize(
                torch.mv(torch.t(w.view(height, -1).data), u.data)
            )
            u.data = self._l2normalize(
                torch.mv(w.view(height, -1).data, v.data)
            )
        
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))
    
    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


def add_spectral_norm(module: nn.Module) -> nn.Module:
    """Add spectral normalization to a module"""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        return SpectralNorm(module)
    return module


class DiscriminatorBlock(nn.Module):
    """Discriminator residual block with optional spectral normalization"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: bool = True,
        use_spectral_norm: bool = True
    ):
        super().__init__()
        
        self.downsample = downsample
        
        # Main path
        conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        if use_spectral_norm:
            conv1 = add_spectral_norm(conv1)
            conv2 = add_spectral_norm(conv2)
        
        layers = [
            conv1,
            nn.LeakyReLU(0.2, inplace=True),
            conv2,
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        if downsample:
            layers.append(nn.AvgPool2d(2))
        
        self.main = nn.Sequential(*layers)
        
        # Skip connection
        skip = []
        if downsample:
            skip.append(nn.AvgPool2d(2))
        if in_channels != out_channels:
            conv_skip = nn.Conv2d(in_channels, out_channels, 1)
            if use_spectral_norm:
                conv_skip = add_spectral_norm(conv_skip)
            skip.append(conv_skip)
        
        self.skip = nn.Sequential(*skip) if skip else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x) + self.skip(x)


class Discriminator(nn.Module):
    """
    Discriminator Network for Vein Image Authentication
    Uses progressive downsampling with spectral normalization
    Outputs both real/fake classification and feature maps for matching
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = None,
        filters: list = None,
        use_spectral_norm: bool = None,
        dropout: float = None,
        input_channels: int = 1
    ):
        """
        Args:
            image_size: Input image size (H, W)
            filters: List of filter sizes for each layer
            use_spectral_norm: Whether to use spectral normalization
            dropout: Dropout rate
            input_channels: Number of input channels (1 for grayscale)
        """
        super().__init__()
        
        # Load from config if not provided
        self.image_size = image_size or config.IMAGE_SIZE
        self.filters = filters or config.DISCRIMINATOR_FILTERS
        self.use_spectral_norm = (use_spectral_norm if use_spectral_norm is not None 
                                  else config.SPECTRAL_NORM)
        self.dropout = dropout or config.DISCRIMINATOR_DROPOUT
        
        # Initial convolution
        conv_init = nn.Conv2d(input_channels, self.filters[0], 3, padding=1)
        if self.use_spectral_norm:
            conv_init = add_spectral_norm(conv_init)
        
        self.init_conv = nn.Sequential(
            conv_init,
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Progressive downsampling blocks
        self.blocks = nn.ModuleList()
        
        for i in range(len(self.filters) - 1):
            in_filters = self.filters[i]
            out_filters = self.filters[i + 1]
            
            block = DiscriminatorBlock(
                in_filters,
                out_filters,
                downsample=True,
                use_spectral_norm=self.use_spectral_norm
            )
            self.blocks.append(block)
        
        # Feature extraction head (for feature matching loss)
        self.feature_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(self.filters[-1] * 16, 512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Classification head (real/fake)
        fc_class = nn.Linear(512, 1)
        if self.use_spectral_norm:
            fc_class = add_spectral_norm(fc_class)
        
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            fc_class
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor or Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: Input image (B, C, H, W)
            return_features: Whether to return intermediate features
        
        Returns:
            If return_features=False: Classification logits (B, 1)
            If return_features=True: (logits, [feature_maps])
        """
        features = []
        
        # Initial convolution
        x = self.init_conv(x)
        if return_features:
            features.append(x)
        
        # Progressive downsampling
        for block in self.blocks:
            x = block(x)
            if return_features:
                features.append(x)
        
        # Extract features
        feat = self.feature_head(x)
        if return_features:
            features.append(feat)
        
        # Classification
        out = self.classifier(feat)
        
        if return_features:
            return out, features
        return out
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature representation for verification
        
        Args:
            x: Input image
        
        Returns:
            Feature vector
        """
        x = self.init_conv(x)
        
        for block in self.blocks:
            x = block(x)
        
        feat = self.feature_head(x)
        return feat


class PatchDiscriminator(nn.Module):
    """
    PatchGAN Discriminator
    Outputs a 2D map instead of single value
    Better for local texture discrimination
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        filters: int = 64,
        n_layers: int = 3,
        use_spectral_norm: bool = True
    ):
        super().__init__()
        
        layers = []
        
        # First layer
        conv = nn.Conv2d(input_channels, filters, 4, stride=2, padding=1)
        if use_spectral_norm:
            conv = add_spectral_norm(conv)
        layers.extend([conv, nn.LeakyReLU(0.2, inplace=True)])
        
        # Middle layers
        in_filters = filters
        for i in range(1, n_layers):
            out_filters = min(filters * (2 ** i), 512)
            conv = nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)
            if use_spectral_norm:
                conv = add_spectral_norm(conv)
            
            layers.extend([
                conv,
                nn.BatchNorm2d(out_filters),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            in_filters = out_filters
        
        # Output layer
        conv = nn.Conv2d(in_filters, 1, 4, padding=1)
        if use_spectral_norm:
            conv = add_spectral_norm(conv)
        layers.append(conv)
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image
        
        Returns:
            Patch-wise predictions
        """
        return self.model(x)


def test_discriminator():
    """Test discriminator forward pass"""
    print("Testing Discriminator...")
    
    # Create discriminator
    disc = Discriminator()
    
    # Test forward pass
    batch_size = 4
    fake_images = torch.randn(batch_size, 1, *config.IMAGE_SIZE)
    
    with torch.no_grad():
        # Standard forward
        logits = disc(fake_images)
        print(f"Input shape: {fake_images.shape}")
        print(f"Output logits shape: {logits.shape}")
        
        # Forward with features
        logits, features = disc(fake_images, return_features=True)
        print(f"Number of feature maps: {len(features)}")
        for i, feat in enumerate(features):
            print(f"  Feature {i} shape: {feat.shape}")
        
        # Extract features only
        feat_vec = disc.extract_features(fake_images)
        print(f"Feature vector shape: {feat_vec.shape}")
    
    # Count parameters
    num_params = sum(p.numel() for p in disc.parameters())
    print(f"Total parameters: {num_params:,}")
    
    return disc


if __name__ == "__main__":
    test_discriminator()