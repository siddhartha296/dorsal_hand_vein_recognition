"""
Comprehensive Loss Functions for XAI-Enhanced Vein GAN
Includes adversarial, perceptual, feature matching, and regularization losses
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights
from typing import Tuple, Optional, Dict, List
import numpy as np

from config import config


class GANLosses:
    """
    Comprehensive collection of GAN loss functions
    """
    
    def __init__(self, device='cuda', loss_type='vanilla'):
        """
        Args:
            device: Device to run on
            loss_type: Type of adversarial loss
                - 'vanilla': Original GAN loss (BCE)
                - 'lsgan': Least Squares GAN
                - 'wgan': Wasserstein GAN
                - 'wgan-gp': WGAN with Gradient Penalty
                - 'hinge': Hinge loss
                - 'relativistic': Relativistic GAN
        """
        self.device = device
        self.loss_type = loss_type
        
        # BCE loss for vanilla GAN
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        
        # Load VGG19 for perceptual loss
        self.vgg = self._load_vgg()
        
        # Feature extraction layers for perceptual loss
        self.perceptual_layers = ['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4']
    
    def _load_vgg(self) -> nn.Module:
        """Load pretrained VGG19 for perceptual loss"""
        try:
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:36].to(self.device).eval()
            
            # Freeze parameters
            for param in vgg.parameters():
                param.requires_grad = False
            
            return vgg
        except Exception as e:
            print(f"Warning: Could not load VGG19: {e}")
            return None
    
    # ==================== Adversarial Losses ====================
    
    def get_adversarial_loss(
        self,
        prediction: torch.Tensor,
        target_is_real: bool,
        for_discriminator: bool = True
    ) -> torch.Tensor:
        """
        Get adversarial loss based on loss type
        
        Args:
            prediction: Discriminator output
            target_is_real: Whether target is real or fake
            for_discriminator: Whether this is for D or G
        
        Returns:
            Loss value
        """
        if self.loss_type == 'vanilla':
            return self._vanilla_gan_loss(prediction, target_is_real)
        
        elif self.loss_type == 'lsgan':
            return self._lsgan_loss(prediction, target_is_real)
        
        elif self.loss_type == 'wgan':
            return self._wgan_loss(prediction, target_is_real, for_discriminator)
        
        elif self.loss_type == 'hinge':
            return self._hinge_loss(prediction, target_is_real, for_discriminator)
        
        elif self.loss_type == 'relativistic':
            # Requires both real and fake predictions
            # Will be handled separately
            raise ValueError("Use get_relativistic_loss for relativistic GAN")
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def _vanilla_gan_loss(
        self,
        prediction: torch.Tensor,
        target_is_real: bool
    ) -> torch.Tensor:
        """Original GAN loss with BCE"""
        target = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
        return self.bce(prediction, target)
    
    def _lsgan_loss(
        self,
        prediction: torch.Tensor,
        target_is_real: bool
    ) -> torch.Tensor:
        """Least Squares GAN loss"""
        target = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
        return self.mse(prediction, target)
    
    def _wgan_loss(
        self,
        prediction: torch.Tensor,
        target_is_real: bool,
        for_discriminator: bool
    ) -> torch.Tensor:
        """Wasserstein GAN loss"""
        if for_discriminator:
            # D wants to maximize D(real) - D(fake)
            # Minimize -(D(real) - D(fake))
            return -prediction.mean() if target_is_real else prediction.mean()
        else:
            # G wants to maximize D(fake)
            # Minimize -D(fake)
            return -prediction.mean()
    
    def _hinge_loss(
        self,
        prediction: torch.Tensor,
        target_is_real: bool,
        for_discriminator: bool
    ) -> torch.Tensor:
        """Hinge loss for GANs"""
        if for_discriminator:
            if target_is_real:
                return F.relu(1.0 - prediction).mean()
            else:
                return F.relu(1.0 + prediction).mean()
        else:
            # Generator wants to fool discriminator
            return -prediction.mean()
    
    def get_relativistic_loss(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor,
        for_discriminator: bool = True
    ) -> torch.Tensor:
        """
        Relativistic average GAN loss
        
        Args:
            real_pred: D(real)
            fake_pred: D(fake)
            for_discriminator: Whether for D or G
        
        Returns:
            Loss value
        """
        if for_discriminator:
            # D(real) should be greater than D(fake)
            loss_real = self.bce(
                real_pred - fake_pred.mean(),
                torch.ones_like(real_pred)
            )
            loss_fake = self.bce(
                fake_pred - real_pred.mean(),
                torch.zeros_like(fake_pred)
            )
            return (loss_real + loss_fake) / 2
        else:
            # G wants D(fake) > D(real)
            loss_real = self.bce(
                real_pred - fake_pred.mean(),
                torch.zeros_like(real_pred)
            )
            loss_fake = self.bce(
                fake_pred - real_pred.mean(),
                torch.ones_like(fake_pred)
            )
            return (loss_real + loss_fake) / 2
    
    # ==================== Perceptual Loss ====================
    
    def get_perceptual_loss(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        layers: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        VGG-based perceptual loss to maintain vein structure
        
        Args:
            real_images: Real images (B, C, H, W)
            fake_images: Generated images (B, C, H, W)
            layers: Which VGG layers to use (None = use defaults)
        
        Returns:
            Perceptual loss
        """
        if self.vgg is None:
            # Fallback to L1 loss if VGG not available
            return self.l1(real_images, fake_images)
        
        # Convert grayscale to RGB for VGG
        real_rgb = self._to_rgb(real_images)
        fake_rgb = self._to_rgb(fake_images)
        
        # Normalize to ImageNet statistics
        real_rgb = self._normalize_imagenet(real_rgb)
        fake_rgb = self._normalize_imagenet(fake_rgb)
        
        # Extract features
        real_features = self._extract_vgg_features(real_rgb)
        fake_features = self._extract_vgg_features(fake_rgb)
        
        # Compute loss across layers
        loss = 0
        for real_feat, fake_feat in zip(real_features, fake_features):
            loss += F.l1_loss(real_feat, fake_feat)
        
        return loss / len(real_features)
    
    def _to_rgb(self, images: torch.Tensor) -> torch.Tensor:
        """Convert grayscale to RGB by repeating channels"""
        if images.size(1) == 1:
            return images.repeat(1, 3, 1, 1)
        return images
    
    def _normalize_imagenet(self, images: torch.Tensor) -> torch.Tensor:
        """Normalize images to ImageNet statistics"""
        # Denormalize from [-1, 1] to [0, 1]
        images = (images + 1) / 2
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
        return (images - mean) / std
    
    def _extract_vgg_features(
        self,
        images: torch.Tensor
    ) -> List[torch.Tensor]:
        """Extract features from VGG19"""
        features = []
        x = images
        
        # Layer indices for feature extraction
        # relu1_2: 3, relu2_2: 8, relu3_4: 17, relu4_4: 26
        layer_indices = [3, 8, 17, 26]
        
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in layer_indices:
                features.append(x)
        
        return features
    
    # ==================== Feature Matching Loss ====================
    
    def get_feature_matching_loss(
        self,
        real_features: List[torch.Tensor],
        fake_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Feature matching loss - match statistics of discriminator features
        
        Args:
            real_features: Features from discriminator for real images
            fake_features: Features from discriminator for fake images
        
        Returns:
            Feature matching loss
        """
        loss = 0
        for real_feat, fake_feat in zip(real_features, fake_features):
            loss += F.l1_loss(real_feat.mean(0), fake_feat.mean(0))
        
        return loss / len(real_features)
    
    # ==================== Gradient Penalty ====================
    
    def get_gradient_penalty(
        self,
        discriminator: nn.Module,
        real_samples: torch.Tensor,
        fake_samples: torch.Tensor,
        lambda_gp: float = 10.0
    ) -> torch.Tensor:
        """
        Gradient penalty for WGAN-GP stability
        
        Args:
            discriminator: Discriminator network
            real_samples: Real images
            fake_samples: Fake images
            lambda_gp: Weight for gradient penalty
        
        Returns:
            Gradient penalty loss
        """
        batch_size = real_samples.size(0)
        
        # Random weight for interpolation
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        
        # Interpolate between real and fake
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        
        # Get discriminator output
        d_interpolates = discriminator(interpolates)
        
        # Compute gradients
        fake = torch.ones(d_interpolates.shape, device=self.device, requires_grad=False)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        # Flatten gradients
        gradients = gradients.view(batch_size, -1)
        
        # Compute gradient penalty
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return lambda_gp * gradient_penalty
    
    # ==================== Style Loss ====================
    
    def get_style_loss(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor
    ) -> torch.Tensor:
        """
        Style loss based on Gram matrices
        Helps maintain texture consistency
        
        Args:
            real_images: Real images
            fake_images: Fake images
        
        Returns:
            Style loss
        """
        if self.vgg is None:
            return torch.tensor(0.0, device=self.device)
        
        # Convert to RGB and normalize
        real_rgb = self._normalize_imagenet(self._to_rgb(real_images))
        fake_rgb = self._normalize_imagenet(self._to_rgb(fake_images))
        
        # Extract features
        real_features = self._extract_vgg_features(real_rgb)
        fake_features = self._extract_vgg_features(fake_rgb)
        
        # Compute Gram matrices and loss
        loss = 0
        for real_feat, fake_feat in zip(real_features, fake_features):
            real_gram = self._gram_matrix(real_feat)
            fake_gram = self._gram_matrix(fake_feat)
            loss += F.mse_loss(real_gram, fake_gram)
        
        return loss / len(real_features)
    
    def _gram_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix for style loss"""
        b, c, h, w = features.size()
        features = features.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
    
    # ==================== Identity Loss ====================
    
    def get_identity_loss(
        self,
        real_images: torch.Tensor,
        reconstructed_images: torch.Tensor
    ) -> torch.Tensor:
        """
        Identity loss - reconstruct real images
        Useful for cycle-consistency or autoencoder-like training
        
        Args:
            real_images: Original real images
            reconstructed_images: Reconstructed images
        
        Returns:
            Identity loss
        """
        return self.l1(real_images, reconstructed_images)
    
    # ==================== Contextual Loss ====================
    
    def get_contextual_loss(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        h: float = 0.5
    ) -> torch.Tensor:
        """
        Contextual loss - measures similarity of image patches
        Good for texture preservation
        
        Args:
            real_images: Real images
            fake_images: Fake images
            h: Bandwidth parameter
        
        Returns:
            Contextual loss
        """
        # Extract patches
        real_patches = self._extract_patches(real_images, patch_size=3, stride=1)
        fake_patches = self._extract_patches(fake_images, patch_size=3, stride=1)
        
        # Compute cosine similarity
        real_patches_norm = F.normalize(real_patches, dim=1)
        fake_patches_norm = F.normalize(fake_patches, dim=1)
        
        # Similarity matrix
        similarity = torch.matmul(real_patches_norm, fake_patches_norm.t())
        
        # Contextual loss
        max_sim, _ = torch.max(similarity, dim=1)
        loss = -torch.log(max_sim + 1e-5).mean()
        
        return loss
    
    def _extract_patches(
        self,
        images: torch.Tensor,
        patch_size: int,
        stride: int
    ) -> torch.Tensor:
        """Extract image patches"""
        patches = F.unfold(images, kernel_size=patch_size, stride=stride)
        return patches.transpose(1, 2)
    
    # ==================== Total Variation Loss ====================
    
    def get_tv_loss(self, images: torch.Tensor) -> torch.Tensor:
        """
        Total Variation loss for smoothness
        
        Args:
            images: Input images
        
        Returns:
            TV loss
        """
        # Horizontal differences
        tv_h = torch.abs(images[:, :, 1:, :] - images[:, :, :-1, :]).mean()
        
        # Vertical differences
        tv_w = torch.abs(images[:, :, :, 1:] - images[:, :, :, :-1]).mean()
        
        return tv_h + tv_w
    
    # ==================== Diversity Loss ====================
    
    def get_diversity_loss(
        self,
        fake_images_1: torch.Tensor,
        fake_images_2: torch.Tensor,
        latent_1: torch.Tensor,
        latent_2: torch.Tensor,
        epsilon: float = 1e-8
    ) -> torch.Tensor:
        """
        Diversity loss to prevent mode collapse
        Encourages different latents to produce different outputs
        
        Args:
            fake_images_1: Images from latent 1
            fake_images_2: Images from latent 2
            latent_1: Latent vector 1
            latent_2: Latent vector 2
            epsilon: Small constant for numerical stability
        
        Returns:
            Diversity loss (negative to maximize diversity)
        """
        # Image distance
        image_dist = F.l1_loss(fake_images_1, fake_images_2, reduction='mean')
        
        # Latent distance
        latent_dist = F.l1_loss(latent_1, latent_2, reduction='mean')
        
        # Ratio (we want high image distance for given latent distance)
        diversity = image_dist / (latent_dist + epsilon)
        
        # Return negative to maximize (since we minimize loss)
        return -diversity


class CombinedGANLoss:
    """
    Combined loss calculator with configurable weights
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        loss_type: str = 'vanilla',
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            device: Device to run on
            loss_type: Type of adversarial loss
            weights: Dictionary of loss weights
        """
        self.losses = GANLosses(device, loss_type)
        
        # Default weights
        self.weights = {
            'adversarial': 1.0,
            'perceptual': 10.0,
            'feature_matching': 10.0,
            'style': 100.0,
            'identity': 10.0,
            'tv': 0.01,
            'gradient_penalty': 10.0,
        }
        
        # Update with provided weights
        if weights is not None:
            self.weights.update(weights)
    
    def compute_generator_loss(
        self,
        fake_pred: torch.Tensor,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        real_features: Optional[List[torch.Tensor]] = None,
        fake_features: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total generator loss
        
        Returns:
            (total_loss, loss_dict)
        """
        loss_dict = {}
        
        # Adversarial loss
        adv_loss = self.losses.get_adversarial_loss(
            fake_pred, target_is_real=True, for_discriminator=False
        )
        loss_dict['adversarial'] = adv_loss.item()
        
        # Perceptual loss
        perceptual_loss = self.losses.get_perceptual_loss(real_images, fake_images)
        loss_dict['perceptual'] = perceptual_loss.item()
        
        # Feature matching loss
        if real_features is not None and fake_features is not None:
            fm_loss = self.losses.get_feature_matching_loss(real_features, fake_features)
            loss_dict['feature_matching'] = fm_loss.item()
        else:
            fm_loss = torch.tensor(0.0, device=real_images.device)
            loss_dict['feature_matching'] = 0.0
        
        # Style loss
        style_loss = self.losses.get_style_loss(real_images, fake_images)
        loss_dict['style'] = style_loss.item()
        
        # Total variation loss
        tv_loss = self.losses.get_tv_loss(fake_images)
        loss_dict['tv'] = tv_loss.item()
        
        # Combine losses
        total_loss = (
            self.weights['adversarial'] * adv_loss +
            self.weights['perceptual'] * perceptual_loss +
            self.weights['feature_matching'] * fm_loss +
            self.weights['style'] * style_loss +
            self.weights['tv'] * tv_loss
        )
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def compute_discriminator_loss(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor,
        discriminator: Optional[nn.Module] = None,
        real_samples: Optional[torch.Tensor] = None,
        fake_samples: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total discriminator loss
        
        Returns:
            (total_loss, loss_dict)
        """
        loss_dict = {}
        
        # Real loss
        real_loss = self.losses.get_adversarial_loss(
            real_pred, target_is_real=True, for_discriminator=True
        )
        loss_dict['real'] = real_loss.item()
        
        # Fake loss
        fake_loss = self.losses.get_adversarial_loss(
            fake_pred, target_is_real=False, for_discriminator=True
        )
        loss_dict['fake'] = fake_loss.item()
        
        # Adversarial loss
        adv_loss = (real_loss + fake_loss) / 2
        loss_dict['adversarial'] = adv_loss.item()
        
        # Gradient penalty (for WGAN-GP)
        if discriminator is not None and real_samples is not None and fake_samples is not None:
            gp_loss = self.losses.get_gradient_penalty(
                discriminator, real_samples, fake_samples
            )
            loss_dict['gradient_penalty'] = gp_loss.item()
        else:
            gp_loss = torch.tensor(0.0, device=real_pred.device)
            loss_dict['gradient_penalty'] = 0.0
        
        # Total loss
        total_loss = adv_loss + self.weights['gradient_penalty'] * gp_loss
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


if __name__ == "__main__":
    # Test losses
    print("Testing GAN Losses...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create dummy data
    batch_size = 4
    real_images = torch.randn(batch_size, 1, 256, 256).to(device)
    fake_images = torch.randn(batch_size, 1, 256, 256).to(device)
    real_pred = torch.randn(batch_size, 1).to(device)
    fake_pred = torch.randn(batch_size, 1).to(device)
    
    # Test basic losses
    losses = GANLosses(device)
    
    adv_loss = losses.get_adversarial_loss(real_pred, True)
    print(f"✓ Adversarial loss: {adv_loss.item():.4f}")
    
    perceptual_loss = losses.get_perceptual_loss(real_images, fake_images)
    print(f"✓ Perceptual loss: {perceptual_loss.item():.4f}")
    
    tv_loss = losses.get_tv_loss(fake_images)
    print(f"✓ TV loss: {tv_loss.item():.4f}")
    
    # Test combined loss
    combined = CombinedGANLoss(device)
    total_g_loss, g_dict = combined.compute_generator_loss(
        fake_pred, real_images, fake_images
    )
    print(f"✓ Total G loss: {total_g_loss.item():.4f}")
    
    total_d_loss, d_dict = combined.compute_discriminator_loss(
        real_pred, fake_pred
    )
    print(f"✓ Total D loss: {total_d_loss.item():.4f}")
    
    print("\n✓ All loss tests passed!")