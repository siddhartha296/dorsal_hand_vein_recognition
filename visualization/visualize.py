"""
Visualization utilities for vein images and GAN training metrics
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from pathlib import Path
from config import config

def plot_vein_batch(images, title="Vein Batch", save_path=None, nrow=8):
    """
    Plots a grid of vein images.
    Input images are expected to be in range [-1, 1].
    """
    # Denormalize: [-1, 1] -> [0, 1]
    images = (images + 1) / 2
    grid = make_grid(images.cpu(), nrow=nrow, padding=2, normalize=False)
    np_grid = grid.numpy()
    
    plt.figure(figsize=(12, 8))
    plt.imshow(np.transpose(np_grid, (1, 2, 0)), cmap='gray')
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_training_results(history, save_path=None):
    """
    Plots training and validation loss curves.
    """
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(history['G_loss'], label="G Loss")
    plt.plot(history['D_loss'], label="D Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def compare_real_fake(real_images, fake_images, save_path=None):
    """
    Creates a side-by-side comparison of real and synthetic vein images.
    """
    real_grid = make_grid(real_images[:8], nrow=8, normalize=True)
    fake_grid = make_grid(fake_images[:8], nrow=8, normalize=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 6))
    axes[0].imshow(np.transpose(real_grid.cpu().numpy(), (1, 2, 0)), cmap='gray')
    axes[0].set_title("Real Vein Patterns")
    axes[0].axis('off')
    
    axes[1].imshow(np.transpose(fake_grid.cpu().numpy(), (1, 2, 0)), cmap='gray')
    axes[1].set_title("Synthetic (GAN) Vein Patterns")
    axes[1].axis('off')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()