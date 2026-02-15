"""
Enhanced Grad-CAM Implementation for Vein Authentication
Includes Grad-CAM, Grad-CAM++, and Score-CAM with comprehensive visualization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Optional, List, Tuple, Dict
from pathlib import Path
import warnings

from config import config


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    Produces visual explanations for CNN decisions
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: str,
        device: str = 'cuda'
    ):
        """
        Args:
            model: CNN model to explain
            target_layer: Name of the layer to visualize (e.g., 'layer4', 'conv4')
            device: Device to run on
        """
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Find target layer in model
        target_module = self._find_target_layer()
        
        if target_module is None:
            raise ValueError(
                f"Target layer '{self.target_layer}' not found in model. "
                f"Available layers: {self._get_available_layers()}"
            )
        
        # Register hooks
        target_module.register_forward_hook(forward_hook)
        target_module.register_full_backward_hook(backward_hook)
    
    def _find_target_layer(self) -> Optional[nn.Module]:
        """Find target layer in model by name"""
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                return module
        return None
    
    def _get_available_layers(self) -> List[str]:
        """Get list of available layer names"""
        return [name for name, _ in self.model.named_modules() if name]
    
    def generate_cam(
        self,
        input_image: torch.Tensor,
        target_class: Optional[int] = None,
        eigen_smooth: bool = False
    ) -> Tuple[np.ndarray, torch.Tensor, int]:
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_image: Input image tensor (1, C, H, W)
            target_class: Target class index (if None, use predicted class)
            eigen_smooth: Apply eigen-based smoothing
        
        Returns:
            (cam_heatmap, model_output, target_class)
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        # Determine target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        class_loss = output[0, target_class]
        class_loss.backward()
        
        # Generate CAM
        if self.gradients is None or self.activations is None:
            warnings.warn("Gradients or activations not captured. Check target layer.")
            return np.zeros((input_image.shape[2], input_image.shape[3])), output, target_class
        
        # Calculate weights (global average pooling of gradients)
        weights = self.gradients[0].mean(dim=(1, 2), keepdim=True)
        
        # Weighted combination of activation maps
        cam = (weights * self.activations[0]).sum(dim=0)
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        cam_np = cam.cpu().numpy()
        
        # Apply eigen smoothing if requested
        if eigen_smooth:
            cam_np = self._eigen_smooth(cam_np, self.activations[0])
        
        return cam_np, output, target_class
    
    def _eigen_smooth(
        self,
        cam: np.ndarray,
        activations: torch.Tensor
    ) -> np.ndarray:
        """
        Apply eigen-based smoothing to reduce noise
        
        Args:
            cam: Original CAM
            activations: Activation maps
        
        Returns:
            Smoothed CAM
        """
        # Reshape activations
        acts = activations.cpu().numpy()
        n_channels, h, w = acts.shape
        acts_flat = acts.reshape(n_channels, -1)
        
        # Compute covariance matrix
        cov = np.cov(acts_flat)
        
        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Use top eigenvectors for smoothing
        top_k = min(5, len(eigenvalues))
        top_indices = np.argsort(eigenvalues)[-top_k:]
        
        # Project and reconstruct
        projection = eigenvectors[:, top_indices]
        acts_projected = projection.T @ acts_flat
        acts_reconstructed = projection @ acts_projected
        
        # Compute smoothed CAM
        cam_smooth = acts_reconstructed.mean(axis=0).reshape(h, w)
        cam_smooth = (cam_smooth - cam_smooth.min()) / (cam_smooth.max() - cam_smooth.min() + 1e-8)
        
        return cam_smooth
    
    def visualize(
        self,
        input_image: torch.Tensor,
        cam: np.ndarray,
        save_path: Optional[Path] = None,
        colormap: str = 'jet',
        alpha: float = 0.5,
        show_original: bool = True
    ):
        """
        Visualize Grad-CAM overlay on input image
        
        Args:
            input_image: Original input image
            cam: CAM heatmap
            save_path: Path to save visualization
            colormap: Colormap for heatmap
            alpha: Transparency for overlay
            show_original: Whether to show original image
        """
        # Resize CAM to input image size
        h, w = input_image.shape[-2:]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Convert CAM to RGB heatmap
        heatmap = cm.get_cmap(colormap)(cam_resized)[:, :, :3]
        
        # Convert input image to numpy
        img_np = input_image[0, 0].cpu().numpy()
        img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        img_rgb = np.stack([img_norm] * 3, axis=-1)
        
        # Create overlay
        overlay = alpha * img_rgb + (1 - alpha) * heatmap
        
        # Plot
        if show_original:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(img_norm, cmap='gray')
            axes[0].set_title("Original Image")
            axes[0].axis('off')
            
            # Heatmap
            axes[1].imshow(heatmap)
            axes[1].set_title("Grad-CAM Heatmap")
            axes[1].axis('off')
            
            # Overlay
            axes[2].imshow(overlay)
            axes[2].set_title("Grad-CAM Overlay")
            axes[2].axis('off')
        else:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.imshow(overlay)
            ax.set_title("Grad-CAM Overlay")
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ implementation
    Improved version with better localization for multiple instances
    """
    
    def generate_cam(
        self,
        input_image: torch.Tensor,
        target_class: Optional[int] = None,
        eigen_smooth: bool = False
    ) -> Tuple[np.ndarray, torch.Tensor, int]:
        """Generate Grad-CAM++ heatmap"""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        # Determine target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        class_loss = output[0, target_class]
        class_loss.backward(retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            warnings.warn("Gradients or activations not captured.")
            return np.zeros((input_image.shape[2], input_image.shape[3])), output, target_class
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        # Calculate alpha (Grad-CAM++ weights)
        # Second derivative
        grad_squared = gradients ** 2
        grad_cubed = grad_squared * gradients
        
        # Global sum
        sum_activations = activations.sum(dim=(1, 2), keepdim=True)
        
        # Alpha calculation
        alpha_denom = 2 * grad_squared + sum_activations * grad_cubed + 1e-8
        alpha = grad_squared / alpha_denom
        
        # Weights (alpha * ReLU(gradient))
        weights = (alpha * F.relu(gradients)).sum(dim=(1, 2), keepdim=True)
        
        # Weighted combination
        cam = (weights * activations).sum(dim=0)
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        cam_np = cam.cpu().numpy()
        
        if eigen_smooth:
            cam_np = self._eigen_smooth(cam_np, activations)
        
        return cam_np, output, target_class


class ScoreCAM:
    """
    Score-CAM: Score-Weighted Class Activation Mapping
    Gradient-free alternative to Grad-CAM
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: str,
        device: str = 'cuda'
    ):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.activations = None
        
        self._register_hook()
    
    def _register_hook(self):
        """Register forward hook"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        # Find and register hook
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                return
    
    def generate_cam(
        self,
        input_image: torch.Tensor,
        target_class: Optional[int] = None,
        batch_size: int = 16
    ) -> Tuple[np.ndarray, torch.Tensor, int]:
        """
        Generate Score-CAM heatmap
        
        Args:
            input_image: Input image
            target_class: Target class
            batch_size: Batch size for processing activation masks
        
        Returns:
            (cam_heatmap, model_output, target_class)
        """
        self.model.eval()
        
        # Forward pass to get activations
        with torch.no_grad():
            output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        if self.activations is None:
            warnings.warn("Activations not captured.")
            return np.zeros((input_image.shape[2], input_image.shape[3])), output, target_class
        
        activations = self.activations[0]  # (C, H, W)
        n_channels = activations.shape[0]
        
        # Normalize each activation map
        norm_acts = torch.zeros_like(activations)
        for i in range(n_channels):
            act = activations[i]
            if act.max() > act.min():
                norm_acts[i] = (act - act.min()) / (act.max() - act.min())
        
        # Upsample to input size
        input_size = input_image.shape[-2:]
        upsampled_acts = F.interpolate(
            norm_acts.unsqueeze(0),
            size=input_size,
            mode='bilinear',
            align_corners=False
        )[0]  # (C, H, W)
        
        # Process in batches
        scores = []
        with torch.no_grad():
            for i in range(0, n_channels, batch_size):
                batch_acts = upsampled_acts[i:i+batch_size]
                
                # Create masked inputs
                masked_inputs = input_image * batch_acts.unsqueeze(1)
                
                # Get scores
                batch_output = self.model(masked_inputs)
                batch_scores = F.softmax(batch_output, dim=1)[:, target_class]
                scores.append(batch_scores)
        
        scores = torch.cat(scores)
        
        # Weight activation maps by scores
        weights = scores.unsqueeze(-1).unsqueeze(-1)
        cam = (weights * norm_acts).sum(dim=0)
        
        # Normalize
        cam = F.relu(cam)
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam.cpu().numpy(), output, target_class


class LayerCAM:
    """
    Layer-CAM: Enhanced Grad-CAM with layer-wise relevance
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: str,
        device: str = 'cuda'
    ):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.gradients = None
        self.activations = None
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                return
    
    def generate_cam(
        self,
        input_image: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, torch.Tensor, int]:
        """Generate Layer-CAM heatmap"""
        self.model.eval()
        
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()
        
        if self.gradients is None or self.activations is None:
            warnings.warn("Gradients or activations not captured.")
            return np.zeros((input_image.shape[2], input_image.shape[3])), output, target_class
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        # Layer-CAM: element-wise multiplication instead of global pooling
        cam = (F.relu(gradients) * activations).sum(dim=0)
        
        # Normalize
        cam = F.relu(cam)
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam.cpu().numpy(), output, target_class


class MultiLayerCAM:
    """
    Multi-layer CAM visualization
    Compares activations across multiple layers
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layers: List[str],
        device: str = 'cuda'
    ):
        self.model = model
        self.target_layers = target_layers
        self.device = device
        self.cams = {}
    
    def generate_multilayer_cam(
        self,
        input_image: torch.Tensor,
        target_class: Optional[int] = None,
        method: str = 'gradcam'
    ) -> Dict[str, np.ndarray]:
        """
        Generate CAM for multiple layers
        
        Args:
            input_image: Input image
            target_class: Target class
            method: CAM method ('gradcam', 'gradcam++', 'layercam')
        
        Returns:
            Dictionary mapping layer names to CAM heatmaps
        """
        cams = {}
        
        for layer_name in self.target_layers:
            try:
                if method == 'gradcam':
                    cam_generator = GradCAM(self.model, layer_name, self.device)
                elif method == 'gradcam++':
                    cam_generator = GradCAMPlusPlus(self.model, layer_name, self.device)
                elif method == 'layercam':
                    cam_generator = LayerCAM(self.model, layer_name, self.device)
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                cam, _, _ = cam_generator.generate_cam(input_image, target_class)
                cams[layer_name] = cam
            except Exception as e:
                warnings.warn(f"Failed to generate CAM for layer {layer_name}: {e}")
                continue
        
        return cams
    
    def visualize_multilayer(
        self,
        input_image: torch.Tensor,
        cams: Dict[str, np.ndarray],
        save_path: Optional[Path] = None
    ):
        """Visualize CAMs from multiple layers"""
        n_layers = len(cams)
        fig, axes = plt.subplots(2, (n_layers + 1) // 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (layer_name, cam) in enumerate(cams.items()):
            # Resize CAM
            h, w = input_image.shape[-2:]
            cam_resized = cv2.resize(cam, (w, h))
            
            # Create heatmap
            heatmap = cm.jet(cam_resized)[:, :, :3]
            
            # Plot
            axes[idx].imshow(heatmap)
            axes[idx].set_title(f"Layer: {layer_name}")
            axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(len(cams), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()


def compare_cam_methods(
    model: nn.Module,
    input_image: torch.Tensor,
    target_layer: str,
    target_class: Optional[int] = None,
    save_path: Optional[Path] = None,
    device: str = 'cuda'
):
    """
    Compare different CAM methods side-by-side
    
    Args:
        model: Model to explain
        input_image: Input image
        target_layer: Target layer name
        target_class: Target class (optional)
        save_path: Save path
        device: Device
    """
    methods = {
        'Grad-CAM': GradCAM(model, target_layer, device),
        'Grad-CAM++': GradCAMPlusPlus(model, target_layer, device),
        'Layer-CAM': LayerCAM(model, target_layer, device)
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    img_np = input_image[0, 0].cpu().numpy()
    img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    
    axes[0, 0].imshow(img_norm, cmap='gray')
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # Generate and plot CAMs
    for idx, (method_name, cam_generator) in enumerate(methods.items(), 1):
        cam, _, _ = cam_generator.generate_cam(input_image, target_class)
        
        # Resize
        h, w = input_image.shape[-2:]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Heatmap
        heatmap = cm.jet(cam_resized)[:, :, :3]
        
        # Overlay
        img_rgb = np.stack([img_norm] * 3, axis=-1)
        overlay = 0.5 * img_rgb + 0.5 * heatmap
        
        # Plot heatmap
        row, col = idx // 3, idx % 3
        axes[0, col].imshow(heatmap)
        axes[0, col].set_title(f"{method_name} Heatmap")
        axes[0, col].axis('off')
        
        # Plot overlay
        axes[1, col].imshow(overlay)
        axes[1, col].set_title(f"{method_name} Overlay")
        axes[1, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


if __name__ == "__main__":
    # Test Grad-CAM
    print("Testing Grad-CAM implementations...")
    
    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Conv2d(1, 64, 3, padding=1)
            self.layer2 = nn.Conv2d(64, 128, 3, padding=1)
            self.layer3 = nn.Conv2d(128, 256, 3, padding=1)
            self.layer4 = nn.Conv2d(256, 512, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, 10)
        
        def forward(self, x):
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
            x = F.relu(self.layer3(x))
            x = F.relu(self.layer4(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = DummyModel()
    dummy_input = torch.randn(1, 1, 256, 256)
    
    # Test Grad-CAM
    gradcam = GradCAM(model, 'layer4')
    cam, output, pred_class = gradcam.generate_cam(dummy_input)
    print(f"✓ Grad-CAM: CAM shape = {cam.shape}, Predicted class = {pred_class}")
    
    # Test Grad-CAM++
    gradcam_pp = GradCAMPlusPlus(model, 'layer4')
    cam_pp, _, _ = gradcam_pp.generate_cam(dummy_input)
    print(f"✓ Grad-CAM++: CAM shape = {cam_pp.shape}")
    
    # Test Layer-CAM
    layercam = LayerCAM(model, 'layer4')
    cam_layer, _, _ = layercam.generate_cam(dummy_input)
    print(f"✓ Layer-CAM: CAM shape = {cam_layer.shape}")
    
    print("\n✓ All Grad-CAM tests passed!")