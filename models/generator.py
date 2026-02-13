"""
Generator Network with Attention Mechanism
For XAI-Enhanced GAN for Dorsal Hand Vein Authentication
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from config import config


class SelfAttention(nn.Module):
    """
    Self-Attention mechanism for feature enhancement
    Based on "Self-Attention Generative Adversarial Networks" (SAGAN)
    """
    
    def __init__(self, in_channels: int, num_heads: int = 8):
        """
        Args:
            in_channels: Number of input channels
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Output projection
        self.out = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Learnable scaling parameter
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Output tensor with attention applied (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Compute query, key, value
        q = self.query(x).view(B, self.num_heads, self.head_dim, H * W)  # (B, heads, head_dim, HW)
        k = self.key(x).view(B, self.num_heads, self.head_dim, H * W)    # (B, heads, head_dim, HW)
        v = self.value(x).view(B, self.num_heads, self.head_dim, H * W)  # (B, heads, head_dim, HW)
        
        # Transpose for matrix multiplication
        q = q.permute(0, 1, 3, 2)  # (B, heads, HW, head_dim)
        
        # Compute attention scores
        attention = torch.matmul(q, k) / (self.head_dim ** 0.5)  # (B, heads, HW, HW)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attention, v.permute(0, 1, 3, 2))  # (B, heads, HW, head_dim)
        
        # Reshape and project
        out = out.permute(0, 1, 3, 2).contiguous()  # (B, heads, head_dim, HW)
        out = out.view(B, C, H, W)
        out = self.out(out)
        
        # Residual connection with learnable weight
        return self.gamma * out + x


class ResidualBlock(nn.Module):
    """Residual block with optional attention"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_attention: bool = False,
        num_heads: int = 8
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        
        # Optional attention
        self.attention = SelfAttention(out_channels, num_heads) if use_attention else None
        
        self.activation = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.attention:
            out = self.attention(out)
        
        out = out + identity
        out = self.activation(out)
        
        return out


class Generator(nn.Module):
    """
    Generator Network for Vein Image Synthesis
    Uses progressive upsampling with attention mechanisms
    """
    
    def __init__(
        self,
        latent_dim: int = None,
        image_size: Tuple[int, int] = None,
        filters: list = None,
        use_attention: bool = None,
        num_heads: int = None,
        output_channels: int = 1
    ):
        """
        Args:
            latent_dim: Dimension of latent space
            image_size: Target output image size (H, W)
            filters: List of filter sizes for each layer
            use_attention: Whether to use attention mechanisms
            num_heads: Number of attention heads
            output_channels: Number of output channels (1 for grayscale)
        """
        super().__init__()
        
        # Load from config if not provided
        self.latent_dim = latent_dim or config.LATENT_DIM
        self.image_size = image_size or config.IMAGE_SIZE
        self.filters = filters or config.GENERATOR_FILTERS
        self.use_attention = use_attention if use_attention is not None else config.USE_ATTENTION
        self.num_heads = num_heads or config.ATTENTION_HEADS
        self.output_channels = output_channels
        
        # Calculate initial spatial size
        # We'll start at 4x4 and upsample to target size
        self.init_size = 4
        num_upsamples = 0
        current_size = self.init_size
        target_size = min(self.image_size)
        
        while current_size < target_size:
            current_size *= 2
            num_upsamples += 1
        
        # Initial projection
        self.fc = nn.Linear(
            self.latent_dim,
            self.filters[0] * self.init_size * self.init_size
        )
        
        # Progressive upsampling blocks
        self.blocks = nn.ModuleList()
        
        for i in range(len(self.filters) - 1):
            in_filters = self.filters[i]
            out_filters = self.filters[i + 1]
            
            # Add residual block
            use_attn = self.use_attention and i >= len(self.filters) // 2  # Attention in later layers
            
            block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                ResidualBlock(in_filters, out_filters, use_attn, self.num_heads),
                ResidualBlock(out_filters, out_filters, use_attn, self.num_heads)
            )
            self.blocks.append(block)
        
        # Final output layer
        self.final = nn.Sequential(
            nn.Conv2d(self.filters[-1], self.filters[-1], 3, padding=1),
            nn.BatchNorm2d(self.filters[-1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.filters[-1], output_channels, 3, padding=1),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generate image from latent code
        
        Args:
            z: Latent code (B, latent_dim)
        
        Returns:
            Generated image (B, C, H, W)
        """
        # Project and reshape
        x = self.fc(z)
        x = x.view(-1, self.filters[0], self.init_size, self.init_size)
        
        # Progressive upsampling
        for block in self.blocks:
            x = block(x)
        
        # Final layer
        x = self.final(x)
        
        # Resize to exact target size if needed
        if x.shape[-2:] != self.image_size:
            x = F.interpolate(
                x,
                size=self.image_size,
                mode='bilinear',
                align_corners=False
            )
        
        return x
    
    def get_attention_maps(self, z: torch.Tensor) -> list:
        """
        Extract attention maps for visualization
        
        Args:
            z: Latent code
        
        Returns:
            List of attention maps from each attention layer
        """
        attention_maps = []
        
        # Forward pass with attention extraction
        x = self.fc(z)
        x = x.view(-1, self.filters[0], self.init_size, self.init_size)
        
        for block in self.blocks:
            for module in block.modules():
                if isinstance(module, SelfAttention):
                    # Hook to extract attention
                    with torch.no_grad():
                        B, C, H, W = x.shape
                        q = module.query(x).view(B, module.num_heads, module.head_dim, H * W)
                        k = module.key(x).view(B, module.num_heads, module.head_dim, H * W)
                        
                        q = q.permute(0, 1, 3, 2)
                        attention = torch.matmul(q, k) / (module.head_dim ** 0.5)
                        attention = F.softmax(attention, dim=-1)
                        
                        # Average across heads and reshape
                        attention = attention.mean(dim=1)  # (B, HW, HW)
                        attention = attention.view(B, H, W, H, W).mean(dim=(3, 4))  # Average attention
                        
                        attention_maps.append(attention)
            
            x = block(x)
        
        return attention_maps


def test_generator():
    """Test generator forward pass"""
    print("Testing Generator...")
    
    # Create generator
    gen = Generator()
    
    # Test forward pass
    batch_size = 4
    z = torch.randn(batch_size, config.LATENT_DIM)
    
    with torch.no_grad():
        fake_images = gen(z)
    
    print(f"Input latent shape: {z.shape}")
    print(f"Output image shape: {fake_images.shape}")
    print(f"Output range: [{fake_images.min():.3f}, {fake_images.max():.3f}]")
    
    # Count parameters
    num_params = sum(p.numel() for p in gen.parameters())
    print(f"Total parameters: {num_params:,}")
    
    return gen


if __name__ == "__main__":
    test_generator()