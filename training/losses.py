"""
Custom Loss Functions for XAI-Enhanced Vein GAN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

class GANLosses:
    def __init__(self, device='cuda'):
        self.device = device
        self.bce = nn.BCELoss()
        # Load VGG19 for perceptual loss
        vgg = vgg19(pretrained=True).features[:18].to(device).eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg

    def get_adversarial_loss(self, prediction, target_is_real):
        """Standard BCE loss for adversarial training"""
        target = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
        return self.bce(prediction, target)

    def get_perceptual_loss(self, real_images, fake_images):
        """VGG-based perceptual loss to maintain vein structure"""
        # Convert grayscale to RGB for VGG
        real_rgb = torch.cat([real_images] * 3, dim=1)
        fake_rgb = torch.cat([fake_images] * 3, dim=1)
        
        real_features = self.vgg(real_rgb)
        fake_features = self.vgg(fake_rgb)
        return F.l1_loss(real_features, fake_features)

    def get_gradient_penalty(self, discriminator, real_samples, fake_samples):
        """Gradient penalty for WGAN-GP stability"""
        alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
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
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty