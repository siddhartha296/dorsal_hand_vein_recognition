"""
Comprehensive Visualization Utilities for Vein Images and GAN Training
Enhanced version with advanced plotting, animation, and analysis tools
"""
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np
from torchvision.utils import make_grid
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
import seaborn as sns
from PIL import Image
import cv2

from config import config


# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_vein_batch(
    images: torch.Tensor,
    title: str = "Vein Batch",
    save_path: Optional[Path] = None,
    nrow: int = 8,
    figsize: Tuple[int, int] = (15, 10),
    show_stats: bool = False
):
    """
    Plot a grid of vein images with optional statistics
    
    Args:
        images: Batch of images (B, C, H, W) in range [-1, 1]
        title: Plot title
        save_path: Path to save figure
        nrow: Number of images per row
        figsize: Figure size
        show_stats: Show image statistics
    """
    # Denormalize: [-1, 1] -> [0, 1]
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)
    
    # Create grid
    grid = make_grid(images.cpu(), nrow=nrow, padding=2, normalize=False)
    np_grid = grid.numpy()
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    if np_grid.shape[0] == 1:
        # Grayscale
        ax.imshow(np_grid[0], cmap='gray')
    else:
        # RGB
        ax.imshow(np.transpose(np_grid, (1, 2, 0)))
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')
    
    # Add statistics
    if show_stats:
        stats_text = f"Min: {images.min():.3f} | Max: {images.max():.3f} | Mean: {images.mean():.3f}"
        ax.text(
            0.5, -0.02, stats_text,
            transform=ax.transAxes,
            ha='center', va='top',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.close()


def compare_real_fake(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    save_path: Optional[Path] = None,
    nrow: int = 8,
    titles: Tuple[str, str] = ("Real Vein Patterns", "Synthetic (GAN) Vein Patterns")
):
    """
    Create a side-by-side comparison of real and synthetic vein images
    
    Args:
        real_images: Real images (B, C, H, W)
        fake_images: Generated images (B, C, H, W)
        save_path: Path to save figure
        nrow: Number of images per row
        titles: Titles for real and fake images
    """
    # Denormalize
    real_images = (real_images + 1) / 2
    fake_images = (fake_images + 1) / 2
    
    real_images = torch.clamp(real_images, 0, 1)
    fake_images = torch.clamp(fake_images, 0, 1)
    
    # Create grids
    real_grid = make_grid(real_images[:nrow].cpu(), nrow=nrow, normalize=False, padding=2)
    fake_grid = make_grid(fake_images[:nrow].cpu(), nrow=nrow, normalize=False, padding=2)
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(18, 10))
    
    # Real images
    if real_grid.size(0) == 1:
        axes[0].imshow(real_grid[0], cmap='gray')
    else:
        axes[0].imshow(np.transpose(real_grid.numpy(), (1, 2, 0)))
    axes[0].set_title(titles[0], fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Fake images
    if fake_grid.size(0) == 1:
        axes[1].imshow(fake_grid[0], cmap='gray')
    else:
        axes[1].imshow(np.transpose(fake_grid.numpy(), (1, 2, 0)))
    axes[1].set_title(titles[1], fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.close()


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Plot training and validation loss curves with multiple metrics
    
    Args:
        history: Dictionary of metric histories
        save_path: Path to save figure
        metrics: Which metrics to plot (None = all)
        figsize: Figure size
    """
    if metrics is None:
        # Automatically detect metrics
        metrics = list(history.keys())
    
    # Separate train and validation metrics
    train_metrics = [m for m in metrics if not m.startswith('val_')]
    val_metrics = [m for m in metrics if m.startswith('val_')]
    
    # Create subplots
    n_metrics = len(train_metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot each metric
    for idx, metric in enumerate(train_metrics):
        ax = axes[idx]
        
        # Train curve
        if metric in history:
            ax.plot(history[metric], label='Train', linewidth=2, marker='o', markersize=3)
        
        # Validation curve
        val_metric = f'val_{metric}' if not metric.startswith('val_') else metric
        if val_metric in history:
            ax.plot(history[val_metric], label='Validation', linewidth=2, marker='s', markersize=3)
        
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(train_metrics), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Training History', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.close()


def plot_gan_losses(
    d_losses: List[float],
    g_losses: List[float],
    save_path: Optional[Path] = None,
    window_size: int = 10
):
    """
    Plot discriminator and generator losses with smoothing
    
    Args:
        d_losses: Discriminator losses
        g_losses: Generator losses
        save_path: Path to save figure
        window_size: Window size for moving average smoothing
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Smooth losses
    d_smooth = moving_average(d_losses, window_size)
    g_smooth = moving_average(g_losses, window_size)
    
    # Plot discriminator loss
    axes[0].plot(d_losses, alpha=0.3, label='Raw', color='blue')
    axes[0].plot(d_smooth, linewidth=2, label='Smoothed', color='darkblue')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Discriminator Loss')
    axes[0].set_title('Discriminator Loss Over Time', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot generator loss
    axes[1].plot(g_losses, alpha=0.3, label='Raw', color='red')
    axes[1].plot(g_smooth, linewidth=2, label='Smoothed', color='darkred')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Generator Loss')
    axes[1].set_title('Generator Loss Over Time', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.close()


def plot_loss_components(
    loss_dict: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    title: str = "Loss Components"
):
    """
    Plot individual loss components as stacked area or separate lines
    
    Args:
        loss_dict: Dictionary of loss component histories
        save_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each component
    for name, values in loss_dict.items():
        ax.plot(values, label=name.replace('_', ' ').title(), linewidth=2, marker='o', markersize=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.close()


def plot_latent_interpolation(
    generator,
    z1: torch.Tensor,
    z2: torch.Tensor,
    steps: int = 10,
    save_path: Optional[Path] = None,
    device: str = 'cuda'
):
    """
    Visualize latent space interpolation between two points
    
    Args:
        generator: Generator model
        z1: Starting latent code
        z2: Ending latent code
        steps: Number of interpolation steps
        save_path: Path to save figure
        device: Device to run on
    """
    generator.eval()
    
    # Create interpolation weights
    alphas = torch.linspace(0, 1, steps).to(device)
    
    # Generate interpolated images
    images = []
    with torch.no_grad():
        for alpha in alphas:
            z_interp = (1 - alpha) * z1 + alpha * z2
            img = generator(z_interp)
            images.append(img)
    
    images = torch.cat(images, dim=0)
    
    # Plot
    plot_vein_batch(
        images,
        title=f"Latent Space Interpolation ({steps} steps)",
        save_path=save_path,
        nrow=steps
    )


def plot_image_evolution(
    image_list: List[torch.Tensor],
    epoch_list: List[int],
    save_path: Optional[Path] = None,
    title: str = "Image Evolution During Training"
):
    """
    Plot how generated images evolve during training
    
    Args:
        image_list: List of generated images at different epochs
        epoch_list: Corresponding epoch numbers
        save_path: Path to save figure
        title: Plot title
    """
    n_images = len(image_list)
    fig, axes = plt.subplots(1, n_images, figsize=(3 * n_images, 4))
    
    if n_images == 1:
        axes = [axes]
    
    for idx, (img, epoch) in enumerate(zip(image_list, epoch_list)):
        # Denormalize
        img = (img + 1) / 2
        img = torch.clamp(img, 0, 1)
        
        # Plot
        img_np = img[0, 0].cpu().numpy() if img.dim() == 4 else img.cpu().numpy()
        axes[idx].imshow(img_np, cmap='gray')
        axes[idx].set_title(f"Epoch {epoch}", fontsize=10, fontweight='bold')
        axes[idx].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.close()


def create_training_animation(
    image_folder: Path,
    output_path: Path,
    fps: int = 5,
    pattern: str = "epoch_*.png"
):
    """
    Create animation from saved training images
    
    Args:
        image_folder: Folder containing training images
        output_path: Path to save animation (gif or mp4)
        fps: Frames per second
        pattern: File pattern to match
    """
    # Get all image files
    image_files = sorted(image_folder.glob(pattern))
    
    if not image_files:
        print(f"⚠ No images found in {image_folder} with pattern {pattern}")
        return
    
    # Read first image to get dimensions
    first_img = Image.open(image_files[0])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis('off')
    
    # Initialize image
    im = ax.imshow(first_img)
    
    def update(frame):
        img = Image.open(image_files[frame])
        im.set_array(img)
        ax.set_title(f"Epoch {frame + 1}", fontsize=14, fontweight='bold')
        return [im]
    
    # Create animation
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(image_files),
        interval=1000 // fps,
        blit=True
    )
    
    # Save
    if output_path.suffix == '.gif':
        anim.save(output_path, writer='pillow', fps=fps)
    elif output_path.suffix == '.mp4':
        anim.save(output_path, writer='ffmpeg', fps=fps)
    else:
        raise ValueError(f"Unsupported format: {output_path.suffix}")
    
    plt.close()
    print(f"✓ Animation saved: {output_path}")


def plot_feature_maps(
    feature_maps: torch.Tensor,
    save_path: Optional[Path] = None,
    max_channels: int = 16,
    title: str = "Feature Maps"
):
    """
    Visualize feature maps from a convolutional layer
    
    Args:
        feature_maps: Feature maps (B, C, H, W) or (C, H, W)
        save_path: Path to save figure
        max_channels: Maximum number of channels to display
        title: Plot title
    """
    if feature_maps.dim() == 4:
        feature_maps = feature_maps[0]  # Take first sample
    
    n_channels = min(feature_maps.size(0), max_channels)
    n_cols = 4
    n_rows = (n_channels + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten()
    
    for idx in range(n_channels):
        fm = feature_maps[idx].cpu().numpy()
        
        # Normalize
        fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
        
        axes[idx].imshow(fm, cmap='viridis')
        axes[idx].set_title(f"Channel {idx}", fontsize=9)
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(n_channels, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.close()


def plot_attention_maps(
    attention_maps: List[torch.Tensor],
    save_path: Optional[Path] = None,
    title: str = "Attention Maps"
):
    """
    Visualize attention maps from self-attention layers
    
    Args:
        attention_maps: List of attention maps
        save_path: Path to save figure
        title: Plot title
    """
    n_maps = len(attention_maps)
    fig, axes = plt.subplots(1, n_maps, figsize=(4 * n_maps, 4))
    
    if n_maps == 1:
        axes = [axes]
    
    for idx, attn in enumerate(attention_maps):
        # Take first sample if batch
        if attn.dim() == 3:
            attn = attn[0]
        
        attn_np = attn.cpu().numpy()
        
        # Normalize
        attn_np = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-8)
        
        im = axes[idx].imshow(attn_np, cmap='hot')
        axes[idx].set_title(f"Layer {idx + 1}", fontsize=10, fontweight='bold')
        axes[idx].axis('off')
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.close()


def plot_distribution_comparison(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    save_path: Optional[Path] = None,
    feature_names: Optional[List[str]] = None,
    max_features: int = 6
):
    """
    Compare feature distributions between real and fake images
    
    Args:
        real_features: Features from real images (N, D)
        fake_features: Features from fake images (N, D)
        save_path: Path to save figure
        feature_names: Names of features
        max_features: Maximum features to plot
    """
    n_features = min(real_features.shape[1], max_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()
    
    for idx in range(n_features):
        ax = axes[idx]
        
        # Plot distributions
        ax.hist(
            real_features[:, idx],
            bins=50,
            alpha=0.5,
            label='Real',
            color='blue',
            density=True
        )
        ax.hist(
            fake_features[:, idx],
            bins=50,
            alpha=0.5,
            label='Fake',
            color='red',
            density=True
        )
        
        feature_name = feature_names[idx] if feature_names else f"Feature {idx}"
        ax.set_title(feature_name, fontsize=10, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Feature Distribution Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.close()


def plot_tsne_embedding(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "t-SNE Feature Embedding"
):
    """
    Plot t-SNE embedding of real and fake features
    
    Args:
        real_features: Features from real images
        fake_features: Features from fake images
        save_path: Path to save figure
        title: Plot title
    """
    from sklearn.manifold import TSNE
    
    # Combine features
    all_features = np.vstack([real_features, fake_features])
    labels = np.array(['Real'] * len(real_features) + ['Fake'] * len(fake_features))
    
    # Compute t-SNE
    print("Computing t-SNE embedding...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedding = tsne.fit_transform(all_features)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for label in ['Real', 'Fake']:
        mask = labels == label
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            label=label,
            alpha=0.6,
            s=50,
            edgecolors='k',
            linewidth=0.5
        )
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.close()


def plot_comprehensive_dashboard(
    epoch: int,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    losses: Dict[str, List[float]],
    metrics: Dict[str, float],
    save_path: Optional[Path] = None
):
    """
    Create comprehensive training dashboard with multiple panels
    
    Args:
        epoch: Current epoch
        real_images: Sample real images
        fake_images: Sample fake images
        losses: Dictionary of loss histories
        metrics: Current metrics
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig)
    
    # 1. Real images
    ax1 = fig.add_subplot(gs[0, 0])
    real_grid = make_grid((real_images[:8] + 1) / 2, nrow=4, padding=2)
    ax1.imshow(real_grid[0].cpu().numpy(), cmap='gray')
    ax1.set_title('Real Images', fontweight='bold')
    ax1.axis('off')
    
    # 2. Fake images
    ax2 = fig.add_subplot(gs[0, 1])
    fake_grid = make_grid((fake_images[:8] + 1) / 2, nrow=4, padding=2)
    ax2.imshow(fake_grid[0].cpu().numpy(), cmap='gray')
    ax2.set_title('Generated Images', fontweight='bold')
    ax2.axis('off')
    
    # 3. Metrics table
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    metric_text = f"Epoch: {epoch}\n\n"
    for key, value in metrics.items():
        metric_text += f"{key}: {value:.4f}\n"
    ax3.text(0.1, 0.9, metric_text, transform=ax3.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax3.set_title('Current Metrics', fontweight='bold')
    
    # 4. Generator loss
    ax4 = fig.add_subplot(gs[1, 0])
    if 'g_total' in losses:
        ax4.plot(losses['g_total'], color='red', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.set_title('Generator Loss', fontweight='bold')
        ax4.grid(True, alpha=0.3)
    
    # 5. Discriminator loss
    ax5 = fig.add_subplot(gs[1, 1])
    if 'd_total' in losses:
        ax5.plot(losses['d_total'], color='blue', linewidth=2)
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Loss')
        ax5.set_title('Discriminator Loss', fontweight='bold')
        ax5.grid(True, alpha=0.3)
    
    # 6. Combined losses
    ax6 = fig.add_subplot(gs[1, 2])
    if 'g_total' in losses and 'd_total' in losses:
        ax6.plot(losses['g_total'], label='Generator', color='red', linewidth=2)
        ax6.plot(losses['d_total'], label='Discriminator', color='blue', linewidth=2)
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Loss')
        ax6.set_title('Combined Losses', fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    # 7-9. Loss components
    loss_components = [k for k in losses.keys() if k not in ['g_total', 'd_total']]
    for idx, loss_name in enumerate(loss_components[:3]):
        ax = fig.add_subplot(gs[2, idx])
        ax.plot(losses[loss_name], linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(loss_name.replace('_', ' ').title(), fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Training Dashboard - Epoch {epoch}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.close()


# ==================== Utility Functions ====================

def moving_average(data: List[float], window_size: int) -> np.ndarray:
    """Compute moving average for smoothing"""
    if len(data) < window_size:
        return np.array(data)
    
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size


def save_image_grid(
    images: torch.Tensor,
    save_path: Path,
    nrow: int = 8,
    normalize: bool = True
):
    """Save a grid of images to file"""
    if normalize:
        images = (images + 1) / 2
        images = torch.clamp(images, 0, 1)
    
    grid = make_grid(images, nrow=nrow, padding=2, normalize=False)
    
    # Convert to PIL and save
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr.squeeze() if ndarr.shape[2] == 1 else ndarr)
    im.save(save_path)


def create_comparison_grid(
    images_dict: Dict[str, torch.Tensor],
    save_path: Optional[Path] = None,
    titles: Optional[Dict[str, str]] = None
):
    """
    Create a comparison grid from multiple image sets
    
    Args:
        images_dict: Dictionary mapping names to image tensors
        save_path: Path to save figure
        titles: Custom titles for each set
    """
    n_sets = len(images_dict)
    fig, axes = plt.subplots(n_sets, 1, figsize=(18, 4 * n_sets))
    
    if n_sets == 1:
        axes = [axes]
    
    for idx, (name, images) in enumerate(images_dict.items()):
        # Denormalize
        images = (images + 1) / 2
        images = torch.clamp(images, 0, 1)
        
        # Create grid
        grid = make_grid(images[:8].cpu(), nrow=8, padding=2, normalize=False)
        
        # Plot
        if grid.size(0) == 1:
            axes[idx].imshow(grid[0], cmap='gray')
        else:
            axes[idx].imshow(np.transpose(grid.numpy(), (1, 2, 0)))
        
        title = titles.get(name, name) if titles else name
        axes[idx].set_title(title, fontsize=12, fontweight='bold')
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.close()


if __name__ == "__main__":
    # Test visualization functions
    print("Testing Visualization Functions...")
    
    # Create dummy data
    real_images = torch.randn(16, 1, 256, 256)
    fake_images = torch.randn(16, 1, 256, 256)
    
    # Test batch plot
    test_dir = Path("test_visualizations")
    test_dir.mkdir(exist_ok=True)
    
    plot_vein_batch(
        real_images,
        title="Test Batch",
        save_path=test_dir / "test_batch.png",
        show_stats=True
    )
    print("✓ plot_vein_batch")
    
    # Test comparison
    compare_real_fake(
        real_images,
        fake_images,
        save_path=test_dir / "test_comparison.png"
    )
    print("✓ compare_real_fake")
    
    # Test training curves
    history = {
        'g_loss': np.random.rand(50).tolist(),
        'd_loss': np.random.rand(50).tolist(),
        'val_g_loss': np.random.rand(50).tolist(),
        'val_d_loss': np.random.rand(50).tolist()
    }
    plot_training_curves(
        history,
        save_path=test_dir / "test_curves.png"
    )
    print("✓ plot_training_curves")
    
    print(f"\n✓ All visualization tests passed!")
    print(f"Test images saved in: {test_dir}")