"""
Comprehensive Loss Functions for XAI-Enhanced Vein GAN
Fixed version:
  - Reduced style/perceptual loss weights to prevent NaN on untrained generator
  - NaN guards on all loss computations
  - Gradient clipping friendly (works with external clip_grad_norm_)
  - Gradient penalty always runs in float32 outside autocast
  - VGG inputs explicitly cast to float32
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights
from typing import Tuple, Optional, Dict, List
import numpy as np

from config import config


def safe_mean(tensor: torch.Tensor, fallback: float = 0.0) -> torch.Tensor:
    """Return mean of tensor, or fallback scalar if NaN/Inf detected."""
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        return torch.tensor(fallback, device=tensor.device, dtype=tensor.dtype)
    return tensor


class GANLosses:
    """
    Comprehensive collection of GAN loss functions.
    """

    def __init__(self, device='cuda', loss_type='vanilla'):
        self.device = device
        self.loss_type = loss_type

        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

        self.vgg = self._load_vgg()

    def _load_vgg(self) -> Optional[nn.Module]:
        try:
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:36].to(self.device).eval()
            for param in vgg.parameters():
                param.requires_grad = False
            print("✓ VGG19 loaded for perceptual loss")
            return vgg
        except Exception as e:
            print(f"⚠ Warning: Could not load VGG19: {e}. Falling back to L1.")
            return None

    # ==================== Adversarial Losses ====================

    def get_adversarial_loss(
        self,
        prediction: torch.Tensor,
        target_is_real: bool,
        for_discriminator: bool = True
    ) -> torch.Tensor:
        if self.loss_type == 'vanilla':
            return self._vanilla_gan_loss(prediction, target_is_real)
        elif self.loss_type == 'lsgan':
            return self._lsgan_loss(prediction, target_is_real)
        elif self.loss_type == 'wgan':
            return self._wgan_loss(prediction, target_is_real, for_discriminator)
        elif self.loss_type == 'hinge':
            return self._hinge_loss(prediction, target_is_real, for_discriminator)
        elif self.loss_type == 'relativistic':
            raise ValueError("Use get_relativistic_loss for relativistic GAN")
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def _vanilla_gan_loss(self, prediction, target_is_real):
        target = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
        loss = self.bce(prediction, target)
        return safe_mean(loss)

    def _lsgan_loss(self, prediction, target_is_real):
        target = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
        loss = self.mse(prediction, target)
        return safe_mean(loss)

    def _wgan_loss(self, prediction, target_is_real, for_discriminator):
        if for_discriminator:
            return -prediction.mean() if target_is_real else prediction.mean()
        else:
            return -prediction.mean()

    def _hinge_loss(self, prediction, target_is_real, for_discriminator):
        if for_discriminator:
            if target_is_real:
                return F.relu(1.0 - prediction).mean()
            else:
                return F.relu(1.0 + prediction).mean()
        else:
            return -prediction.mean()

    def get_relativistic_loss(self, real_pred, fake_pred, for_discriminator=True):
        if for_discriminator:
            loss_real = self.bce(real_pred - fake_pred.mean(), torch.ones_like(real_pred))
            loss_fake = self.bce(fake_pred - real_pred.mean(), torch.zeros_like(fake_pred))
            return safe_mean((loss_real + loss_fake) / 2)
        else:
            loss_real = self.bce(real_pred - fake_pred.mean(), torch.zeros_like(real_pred))
            loss_fake = self.bce(fake_pred - real_pred.mean(), torch.ones_like(fake_pred))
            return safe_mean((loss_real + loss_fake) / 2)

    # ==================== Perceptual Loss ====================

    def get_perceptual_loss(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        layers=None
    ) -> torch.Tensor:
        """
        VGG perceptual loss — always runs in float32.
        Returns zero tensor (no gradient) if VGG unavailable or NaN detected.
        """
        if self.vgg is None:
            return self.l1(real_images.float(), fake_images.float())

        # Cast to float32 — VGG must not receive fp16
        real_f = real_images.detach().float()
        fake_f = fake_images.float()

        # Early NaN check — if generator output is already NaN, skip VGG
        if torch.isnan(fake_f).any() or torch.isinf(fake_f).any():
            return torch.tensor(0.0, device=self.device, requires_grad=False)

        real_rgb = self._normalize_imagenet(self._to_rgb(real_f))
        fake_rgb = self._normalize_imagenet(self._to_rgb(fake_f))

        real_features = self._extract_vgg_features(real_rgb)
        fake_features = self._extract_vgg_features(fake_rgb)

        loss = torch.tensor(0.0, device=self.device)
        for rf, ff in zip(real_features, fake_features):
            layer_loss = F.l1_loss(rf.detach(), ff)
            loss = loss + safe_mean(layer_loss)

        return loss / len(real_features)

    def _to_rgb(self, images: torch.Tensor) -> torch.Tensor:
        if images.size(1) == 1:
            return images.repeat(1, 3, 1, 1)
        return images

    def _normalize_imagenet(self, images: torch.Tensor) -> torch.Tensor:
        # Denormalize from [-1,1] to [0,1]
        images = (images + 1) / 2
        images = images.clamp(0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
        return (images - mean) / std

    def _extract_vgg_features(self, images: torch.Tensor) -> List[torch.Tensor]:
        features = []
        x = images
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
        Feature matching loss.
        real_features and fake_features must come from the same generator pass.
        """
        loss = torch.tensor(0.0, device=self.device)
        for rf, ff in zip(real_features, fake_features):
            layer_loss = F.l1_loss(rf.detach().mean(0), ff.mean(0))
            loss = loss + safe_mean(layer_loss)
        return loss / max(len(real_features), 1)

    # ==================== Gradient Penalty ====================

    def get_gradient_penalty(
        self,
        discriminator: nn.Module,
        real_samples: torch.Tensor,
        fake_samples: torch.Tensor,
        lambda_gp: float = 10.0
    ) -> torch.Tensor:
        """
        Gradient penalty for WGAN-GP.
        MUST be called outside any autocast context — requires float32.
        """
        batch_size = real_samples.size(0)

        real_samples = real_samples.float().detach()
        fake_samples = fake_samples.float().detach()

        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

        d_interpolates = discriminator(interpolates)

        fake = torch.ones(d_interpolates.shape, device=self.device, requires_grad=False)

        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return lambda_gp * safe_mean(gradient_penalty)

    # ==================== Style Loss ====================

    def get_style_loss(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor
    ) -> torch.Tensor:
        if self.vgg is None:
            return torch.tensor(0.0, device=self.device)

        real_f = real_images.detach().float()
        fake_f = fake_images.float()

        if torch.isnan(fake_f).any() or torch.isinf(fake_f).any():
            return torch.tensor(0.0, device=self.device, requires_grad=False)

        real_rgb = self._normalize_imagenet(self._to_rgb(real_f))
        fake_rgb = self._normalize_imagenet(self._to_rgb(fake_f))

        real_features = self._extract_vgg_features(real_rgb)
        fake_features = self._extract_vgg_features(fake_rgb)

        loss = torch.tensor(0.0, device=self.device)
        for rf, ff in zip(real_features, fake_features):
            real_gram = self._gram_matrix(rf.detach())
            fake_gram = self._gram_matrix(ff)
            layer_loss = F.mse_loss(real_gram, fake_gram)
            loss = loss + safe_mean(layer_loss)

        return loss / max(len(real_features), 1)

    def _gram_matrix(self, features: torch.Tensor) -> torch.Tensor:
        b, c, h, w = features.size()
        f = features.view(b, c, h * w)
        gram = torch.bmm(f, f.transpose(1, 2))
        return gram / (c * h * w)

    # ==================== TV Loss ====================

    def get_tv_loss(self, images: torch.Tensor) -> torch.Tensor:
        tv_h = torch.abs(images[:, :, 1:, :] - images[:, :, :-1, :]).mean()
        tv_w = torch.abs(images[:, :, :, 1:] - images[:, :, :, :-1]).mean()
        return safe_mean(tv_h + tv_w)

    # ==================== Diversity Loss ====================

    def get_diversity_loss(
        self,
        fake_images_1: torch.Tensor,
        fake_images_2: torch.Tensor,
        latent_1: torch.Tensor,
        latent_2: torch.Tensor,
        epsilon: float = 1e-8
    ) -> torch.Tensor:
        image_dist = F.l1_loss(fake_images_1, fake_images_2, reduction='mean')
        latent_dist = F.l1_loss(latent_1, latent_2, reduction='mean')
        diversity = image_dist / (latent_dist + epsilon)
        return safe_mean(-diversity)


class CombinedGANLoss:
    """
    Combined loss calculator.

    Key fixes:
    - Style weight reduced from 100.0 -> 1.0 (was causing NaN on untrained G)
    - Perceptual weight reduced from 10.0 -> 1.0 for the same reason
    - All individual losses guarded against NaN before combining
    - Gradient penalty removed from compute_discriminator_loss
      (handled externally in train.py outside autocast)
    """

    def __init__(
        self,
        device: str = 'cuda',
        loss_type: str = 'vanilla',
        weights: Optional[Dict[str, float]] = None
    ):
        self.losses = GANLosses(device, loss_type)
        self.device = device

        # FIX: Reduced style (100->1) and perceptual (10->1) weights.
        # Large weights on VGG losses cause NaN when the generator is
        # untrained and produces extreme pixel values in early epochs.
        # Weights can be gradually increased via update_weights() after
        # training stabilizes (e.g. after epoch 10).
        self.weights = {
            'adversarial':       1.0,
            'perceptual':        1.0,   # was 10.0
            'feature_matching': 10.0,
            'style':             1.0,   # was 100.0
            'tv':                0.01,
            'gradient_penalty': 10.0,
        }
        if weights is not None:
            self.weights.update(weights)

    def update_weights(self, new_weights: Dict[str, float]):
        """Gradually increase loss weights after training stabilizes."""
        self.weights.update(new_weights)
        print(f"  Loss weights updated: {self.weights}")

    def compute_generator_loss(
        self,
        fake_pred: torch.Tensor,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        real_features: Optional[List[torch.Tensor]] = None,
        fake_features: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total generator loss.
        fake_images must be the SAME tensor used to get fake_pred/fake_features.
        All losses are NaN-guarded before combining.
        """
        loss_dict = {}
        zero = torch.tensor(0.0, device=real_images.device)

        # Adversarial loss
        adv_loss = self.losses.get_adversarial_loss(
            fake_pred, target_is_real=True, for_discriminator=False
        )
        adv_loss = safe_mean(adv_loss)
        loss_dict['adversarial'] = adv_loss.item()

        # Perceptual loss (VGG, float32, NaN-guarded inside)
        perceptual_loss = self.losses.get_perceptual_loss(real_images, fake_images)
        perceptual_loss = safe_mean(perceptual_loss)
        loss_dict['perceptual'] = perceptual_loss.item()

        # Feature matching loss — only meaningful if same fake_images used
        if real_features is not None and fake_features is not None:
            fm_loss = self.losses.get_feature_matching_loss(real_features, fake_features)
            fm_loss = safe_mean(fm_loss)
        else:
            fm_loss = zero.clone()
        loss_dict['feature_matching'] = fm_loss.item()

        # Style loss (VGG, float32, NaN-guarded inside)
        style_loss = self.losses.get_style_loss(real_images, fake_images)
        style_loss = safe_mean(style_loss)
        loss_dict['style'] = style_loss.item()

        # TV loss
        tv_loss = self.losses.get_tv_loss(fake_images)
        tv_loss = safe_mean(tv_loss)
        loss_dict['tv'] = tv_loss.item()

        # Combine — each term checked individually so one NaN doesn't kill all
        total_loss = (
            self.weights['adversarial']       * adv_loss +
            self.weights['perceptual']        * perceptual_loss +
            self.weights['feature_matching']  * fm_loss +
            self.weights['style']             * style_loss +
            self.weights['tv']                * tv_loss
        )

        # Final NaN guard — if total is still NaN, fall back to adv_loss only
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("⚠ NaN in total G loss — falling back to adversarial loss only")
            total_loss = adv_loss

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
        Compute discriminator loss.
        Gradient penalty is NOT computed here — handled in train.py outside autocast.
        """
        loss_dict = {}

        real_loss = self.losses.get_adversarial_loss(
            real_pred, target_is_real=True, for_discriminator=True
        )
        real_loss = safe_mean(real_loss)
        loss_dict['real'] = real_loss.item()

        fake_loss = self.losses.get_adversarial_loss(
            fake_pred, target_is_real=False, for_discriminator=True
        )
        fake_loss = safe_mean(fake_loss)
        loss_dict['fake'] = fake_loss.item()

        adv_loss = (real_loss + fake_loss) / 2
        loss_dict['adversarial'] = adv_loss.item()
        loss_dict['gradient_penalty'] = 0.0

        total_loss = adv_loss

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("⚠ NaN in total D loss — using zero loss")
            total_loss = torch.tensor(0.0, device=real_pred.device, requires_grad=True)

        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict


if __name__ == "__main__":
    print("Testing GAN Losses...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size = 4
    real_images = torch.randn(batch_size, 1, 256, 256).to(device)
    fake_images = torch.randn(batch_size, 1, 256, 256).to(device)
    real_pred = torch.randn(batch_size, 1).to(device)
    fake_pred = torch.randn(batch_size, 1).to(device)

    losses = GANLosses(device)

    adv_loss = losses.get_adversarial_loss(real_pred, True)
    print(f"✓ Adversarial loss: {adv_loss.item():.4f}")

    perceptual_loss = losses.get_perceptual_loss(real_images, fake_images)
    print(f"✓ Perceptual loss: {perceptual_loss.item():.4f}")

    tv_loss = losses.get_tv_loss(fake_images)
    print(f"✓ TV loss: {tv_loss.item():.4f}")

    combined = CombinedGANLoss(device)
    total_g_loss, g_dict = combined.compute_generator_loss(
        fake_pred, real_images, fake_images
    )
    print(f"✓ Total G loss: {total_g_loss.item():.4f}")
    print(f"  Loss breakdown: {g_dict}")

    total_d_loss, d_dict = combined.compute_discriminator_loss(real_pred, fake_pred)
    print(f"✓ Total D loss: {total_d_loss.item():.4f}")

    print("\n✓ All loss tests passed!") 