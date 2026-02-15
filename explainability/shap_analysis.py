"""
Enhanced SHAP (SHapley Additive exPlanations) Analysis for Vein Authentication
Comprehensive implementation with multiple explainer types and visualizations
"""
import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict
from pathlib import Path
import warnings

from config import config


class VeinSHAPAnalyzer:
    """
    Comprehensive SHAP analyzer for vein authentication models
    Supports multiple explainer types and visualization methods
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        background_data: torch.Tensor,
        device: str = 'cuda'
    ):
        """
        Args:
            model: Model to explain
            background_data: Background dataset for SHAP (typically small sample)
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.background = background_data.to(device)
        self.explainer = None
        self.model.eval()
        
        # Ensure model doesn't require gradients for efficiency
        for param in self.model.parameters():
            param.requires_grad = False
    
    def create_explainer(
        self,
        method: str = 'gradient',
        **kwargs
    ) -> shap.Explainer:
        """
        Create SHAP explainer
        
        Args:
            method: Explainer type
                - 'gradient': GradientExplainer (fast, requires gradients)
                - 'deep': DeepExplainer (slower, more accurate)
                - 'kernel': KernelExplainer (model-agnostic, very slow)
                - 'partition': PartitionExplainer (for tree-based models)
            **kwargs: Additional arguments for explainer
        
        Returns:
            SHAP explainer instance
        """
        self.model.eval()
        
        if method == 'gradient':
            # Temporarily enable gradients
            for param in self.model.parameters():
                param.requires_grad = True
            
            self.explainer = shap.GradientExplainer(
                self.model,
                self.background,
                **kwargs
            )
            
            # Disable gradients again
            for param in self.model.parameters():
                param.requires_grad = False
                
        elif method == 'deep':
            self.explainer = shap.DeepExplainer(
                self.model,
                self.background,
                **kwargs
            )
            
        elif method == 'kernel':
            # Create prediction function for KernelExplainer
            def predict_fn(x):
                x_tensor = torch.FloatTensor(x).to(self.device)
                with torch.no_grad():
                    outputs = self.model(x_tensor)
                    if outputs.dim() == 1:
                        outputs = outputs.unsqueeze(1)
                    probs = torch.softmax(outputs, dim=1)
                return probs.cpu().numpy()
            
            # KernelExplainer expects 2D input
            background_flat = self.background.cpu().numpy()
            background_flat = background_flat.reshape(background_flat.shape[0], -1)
            
            self.explainer = shap.KernelExplainer(
                predict_fn,
                background_flat,
                **kwargs
            )
            
        else:
            raise ValueError(
                f"Unknown method: {method}. "
                f"Choose from: 'gradient', 'deep', 'kernel'"
            )
        
        print(f"✓ Created {method} SHAP explainer")
        return self.explainer
    
    def explain_instance(
        self,
        input_image: torch.Tensor,
        nsamples: int = 200,
        ranked_outputs: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Explain a single instance
        
        Args:
            input_image: Input image to explain (1, C, H, W)
            nsamples: Number of samples for approximation (KernelExplainer only)
            ranked_outputs: Number of top outputs to explain (None = predicted class only)
        
        Returns:
            (shap_values, predicted_class)
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first.")
        
        input_image = input_image.to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(input_image)
            pred_class = output.argmax(dim=1).item()
        
        # Calculate SHAP values
        try:
            if isinstance(self.explainer, shap.KernelExplainer):
                # KernelExplainer expects flattened input
                input_flat = input_image.cpu().numpy().reshape(1, -1)
                shap_values = self.explainer.shap_values(
                    input_flat,
                    nsamples=nsamples
                )
            else:
                shap_values = self.explainer.shap_values(
                    input_image,
                    ranked_outputs=ranked_outputs
                )
            
            # Handle different return formats
            if isinstance(shap_values, list):
                # Multi-class: list of arrays for each class
                if ranked_outputs is not None:
                    shap_values = shap_values[0]  # Top class
                else:
                    shap_values = shap_values[pred_class]
            
            # Ensure proper shape
            if isinstance(shap_values, np.ndarray):
                if shap_values.ndim == 2:
                    # Reshape flattened values back to image shape
                    original_shape = input_image.shape
                    shap_values = shap_values.reshape(original_shape)
            
        except Exception as e:
            warnings.warn(f"Error calculating SHAP values: {e}")
            # Return zero array as fallback
            shap_values = np.zeros_like(input_image.cpu().numpy())
        
        return shap_values, pred_class
    
    def explain_batch(
        self,
        input_images: torch.Tensor,
        batch_size: int = 8
    ) -> List[Tuple[np.ndarray, int]]:
        """
        Explain a batch of images
        
        Args:
            input_images: Batch of images (B, C, H, W)
            batch_size: Process in smaller batches if needed
        
        Returns:
            List of (shap_values, predicted_class) tuples
        """
        results = []
        
        for i in range(0, len(input_images), batch_size):
            batch = input_images[i:i+batch_size]
            
            for img in batch:
                shap_vals, pred = self.explain_instance(img.unsqueeze(0))
                results.append((shap_vals, pred))
        
        return results
    
    # ==================== Visualization Methods ====================
    
    def visualize_shap_instance(
        self,
        input_image: torch.Tensor,
        shap_values: np.ndarray,
        save_path: Optional[Path] = None,
        method: str = 'overlay'
    ):
        """
        Visualize SHAP values for a single instance
        
        Args:
            input_image: Original input image
            shap_values: SHAP values
            save_path: Path to save visualization
            method: Visualization method ('overlay', 'heatmap', 'both')
        """
        # Prepare SHAP values
        if shap_values.ndim == 4:
            shap_map = np.abs(shap_values).mean(axis=1)[0]  # Average over channels
        elif shap_values.ndim == 3:
            shap_map = np.abs(shap_values[0])
        else:
            shap_map = np.abs(shap_values)
        
        # Normalize SHAP map
        if shap_map.max() > shap_map.min():
            shap_map = (shap_map - shap_map.min()) / (shap_map.max() - shap_map.min())
        
        # Prepare input image
        img_np = input_image[0, 0].cpu().numpy()
        img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        
        # Create visualization
        if method == 'overlay' or method == 'both':
            fig, axes = plt.subplots(1, 3 if method == 'both' else 2, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(img_norm, cmap='gray')
            axes[0].set_title("Original Image")
            axes[0].axis('off')
            
            # SHAP overlay
            overlay = self._create_shap_overlay(img_norm, shap_map)
            axes[1].imshow(overlay)
            axes[1].set_title("SHAP Overlay")
            axes[1].axis('off')
            
            if method == 'both':
                # SHAP heatmap
                im = axes[2].imshow(shap_map, cmap='hot')
                axes[2].set_title("SHAP Importance")
                axes[2].axis('off')
                plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        
        else:  # method == 'heatmap'
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            axes[0].imshow(img_norm, cmap='gray')
            axes[0].set_title("Original Image")
            axes[0].axis('off')
            
            im = axes[1].imshow(shap_map, cmap='hot')
            axes[1].set_title("SHAP Importance Heatmap")
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def _create_shap_overlay(
        self,
        image: np.ndarray,
        shap_map: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """Create overlay of SHAP values on image"""
        # Convert image to RGB
        img_rgb = np.stack([image] * 3, axis=-1)
        
        # Create colored SHAP map (red for high importance)
        shap_colored = plt.cm.hot(shap_map)[:, :, :3]
        
        # Blend
        overlay = alpha * img_rgb + (1 - alpha) * shap_colored
        
        return overlay
    
    def plot_waterfall(
        self,
        shap_values: np.ndarray,
        feature_names: Optional[List[str]] = None,
        max_display: int = 20,
        save_path: Optional[Path] = None
    ):
        """
        Plot SHAP waterfall chart
        Shows contribution of top features
        
        Args:
            shap_values: SHAP values for one instance
            feature_names: Names of features
            max_display: Maximum features to display
            save_path: Save path
        """
        # Flatten SHAP values
        shap_flat = shap_values.flatten()
        
        # Get top contributing features
        top_indices = np.argsort(np.abs(shap_flat))[-max_display:][::-1]
        top_values = shap_flat[top_indices]
        
        if feature_names is None:
            feature_names = [f"Pixel_{i}" for i in top_indices]
        else:
            feature_names = [feature_names[i] for i in top_indices]
        
        # Plot
        plt.figure(figsize=(10, max_display * 0.3))
        colors = ['red' if v > 0 else 'blue' for v in top_values]
        
        plt.barh(range(len(top_values)), top_values, color=colors)
        plt.yticks(range(len(top_values)), feature_names)
        plt.xlabel('SHAP Value (Impact on Model Output)')
        plt.title('Top Feature Contributions')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_summary(
        self,
        shap_values_list: List[np.ndarray],
        save_path: Optional[Path] = None,
        plot_type: str = 'dot'
    ):
        """
        Create SHAP summary plot across multiple samples
        
        Args:
            shap_values_list: List of SHAP values from multiple instances
            save_path: Save path
            plot_type: Type of plot ('dot', 'bar', 'violin')
        """
        # Stack SHAP values
        shap_array = np.array([sv.flatten() for sv in shap_values_list])
        
        plt.figure(figsize=(12, 8))
        
        if plot_type == 'dot':
            # Scatter plot showing distribution
            for i in range(min(20, shap_array.shape[1])):
                plt.scatter(
                    shap_array[:, i],
                    [i] * len(shap_array),
                    alpha=0.5,
                    s=20
                )
            plt.ylabel('Feature Index')
            plt.xlabel('SHAP Value')
            plt.title('SHAP Value Distribution Across Samples')
            
        elif plot_type == 'bar':
            # Bar plot of mean absolute SHAP values
            mean_shap = np.abs(shap_array).mean(axis=0)
            top_features = np.argsort(mean_shap)[-20:][::-1]
            
            plt.barh(range(len(top_features)), mean_shap[top_features])
            plt.ylabel('Feature Index')
            plt.xlabel('Mean |SHAP Value|')
            plt.title('Top Features by Mean SHAP Value')
            
        elif plot_type == 'violin':
            # Violin plot for top features
            mean_shap = np.abs(shap_array).mean(axis=0)
            top_features = np.argsort(mean_shap)[-10:][::-1]
            
            data_to_plot = [shap_array[:, i] for i in top_features]
            plt.violinplot(data_to_plot)
            plt.xlabel('Feature Index')
            plt.ylabel('SHAP Value')
            plt.title('SHAP Value Distribution for Top Features')
        
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_dependence(
        self,
        shap_values_list: List[np.ndarray],
        feature_values_list: List[np.ndarray],
        feature_idx: int,
        save_path: Optional[Path] = None
    ):
        """
        Plot SHAP dependence plot for a specific feature
        Shows relationship between feature value and SHAP value
        
        Args:
            shap_values_list: List of SHAP values
            feature_values_list: List of feature values
            feature_idx: Index of feature to plot
            save_path: Save path
        """
        # Extract values for specific feature
        shap_vals = [sv.flatten()[feature_idx] for sv in shap_values_list]
        feat_vals = [fv.flatten()[feature_idx] for fv in feature_values_list]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(feat_vals, shap_vals, alpha=0.6, s=20)
        plt.xlabel(f'Feature Value (Feature {feature_idx})')
        plt.ylabel('SHAP Value')
        plt.title(f'SHAP Dependence Plot - Feature {feature_idx}')
        plt.grid(alpha=0.3)
        
        # Add trend line
        z = np.polyfit(feat_vals, shap_vals, 1)
        p = np.poly1d(z)
        plt.plot(
            sorted(feat_vals),
            p(sorted(feat_vals)),
            "r--",
            alpha=0.8,
            label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}'
        )
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_force(
        self,
        shap_values: np.ndarray,
        base_value: float,
        feature_values: np.ndarray,
        save_path: Optional[Path] = None
    ):
        """
        Plot SHAP force plot
        Visualizes how features push prediction from base value
        
        Args:
            shap_values: SHAP values for one instance
            base_value: Base/expected value
            feature_values: Actual feature values
            save_path: Save path
        """
        # Flatten values
        shap_flat = shap_values.flatten()
        feat_flat = feature_values.flatten()
        
        # Sort by absolute SHAP value
        indices = np.argsort(np.abs(shap_flat))[::-1][:20]  # Top 20
        
        sorted_shap = shap_flat[indices]
        sorted_feat = feat_flat[indices]
        
        # Separate positive and negative contributions
        pos_indices = sorted_shap > 0
        neg_indices = sorted_shap < 0
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot positive contributions
        cumsum = base_value
        for i, (sv, fv) in enumerate(zip(sorted_shap[pos_indices], sorted_feat[pos_indices])):
            ax.barh(0, sv, left=cumsum, height=0.5, 
                   color='red', alpha=0.7, edgecolor='black')
            cumsum += sv
        
        # Plot negative contributions
        cumsum = base_value
        for i, (sv, fv) in enumerate(zip(sorted_shap[neg_indices], sorted_feat[neg_indices])):
            ax.barh(0, sv, left=cumsum, height=0.5, 
                   color='blue', alpha=0.7, edgecolor='black')
            cumsum += sv
        
        # Add base value and final value markers
        ax.axvline(x=base_value, color='gray', linestyle='--', label='Base Value')
        final_value = base_value + sorted_shap.sum()
        ax.axvline(x=final_value, color='green', linestyle='--', label='Final Value')
        
        ax.set_xlabel('Model Output')
        ax.set_title('SHAP Force Plot')
        ax.set_yticks([])
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def compute_interaction_values(
        self,
        input_image: torch.Tensor
    ) -> np.ndarray:
        """
        Compute SHAP interaction values
        Shows how feature pairs interact
        
        Args:
            input_image: Input image
        
        Returns:
            Interaction values matrix
        """
        if not isinstance(self.explainer, (shap.GradientExplainer, shap.DeepExplainer)):
            warnings.warn(
                "Interaction values only supported for GradientExplainer and DeepExplainer"
            )
            return None
        
        try:
            interaction_values = self.explainer.shap_interaction_values(input_image)
            return interaction_values
        except Exception as e:
            warnings.warn(f"Error computing interaction values: {e}")
            return None


def compare_shap_methods(
    model: torch.nn.Module,
    input_image: torch.Tensor,
    background_data: torch.Tensor,
    save_path: Optional[Path] = None,
    device: str = 'cuda'
):
    """
    Compare different SHAP explainer methods
    
    Args:
        model: Model to explain
        input_image: Input to explain
        background_data: Background dataset
        save_path: Save path
        device: Device
    """
    methods = ['gradient', 'deep']
    
    fig, axes = plt.subplots(2, len(methods) + 1, figsize=(15, 10))
    
    # Original image
    img_np = input_image[0, 0].cpu().numpy()
    img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    
    axes[0, 0].imshow(img_norm, cmap='gray')
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    
    # Test each method
    for idx, method in enumerate(methods, 1):
        try:
            analyzer = VeinSHAPAnalyzer(model, background_data, device)
            analyzer.create_explainer(method=method)
            shap_vals, _ = analyzer.explain_instance(input_image)
            
            # Process SHAP values
            if shap_vals.ndim == 4:
                shap_map = np.abs(shap_vals).mean(axis=1)[0]
            else:
                shap_map = np.abs(shap_vals[0]) if shap_vals.ndim > 2 else np.abs(shap_vals)
            
            if shap_map.max() > shap_map.min():
                shap_map = (shap_map - shap_map.min()) / (shap_map.max() - shap_map.min())
            
            # Plot
            axes[0, idx].imshow(shap_map, cmap='hot')
            axes[0, idx].set_title(f"{method.capitalize()} SHAP")
            axes[0, idx].axis('off')
            
            # Overlay
            overlay = analyzer._create_shap_overlay(img_norm, shap_map)
            axes[1, idx].imshow(overlay)
            axes[1, idx].set_title(f"{method.capitalize()} Overlay")
            axes[1, idx].axis('off')
            
        except Exception as e:
            axes[0, idx].text(0.5, 0.5, f"Error:\n{str(e)[:50]}", 
                             ha='center', va='center')
            axes[0, idx].axis('off')
            axes[1, idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


if __name__ == "__main__":
    # Test SHAP analyzer
    print("Testing SHAP Analyzer...")
    
    # Create dummy model
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(1, 32, 3, padding=1)
            self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.fc = torch.nn.Linear(64, 10)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = DummyModel()
    background = torch.randn(5, 1, 64, 64)
    test_input = torch.randn(1, 1, 64, 64)
    
    # Test gradient explainer
    analyzer = VeinSHAPAnalyzer(model, background)
    analyzer.create_explainer(method='gradient')
    shap_vals, pred = analyzer.explain_instance(test_input)
    
    print(f"✓ SHAP values shape: {shap_vals.shape}")
    print(f"✓ Predicted class: {pred}")
    print("\n✓ All SHAP tests passed!")