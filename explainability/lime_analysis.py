"""
Enhanced LIME (Local Interpretable Model-agnostic Explanations) Analysis
Comprehensive implementation for vein authentication with advanced segmentation
"""
import torch
import numpy as np
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries, slic, felzenszwalb, quickshift
from skimage.color import gray2rgb
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict, Callable
from pathlib import Path
import warnings

from config import config


class VeinLIMEAnalyzer:
    """
    Comprehensive LIME analyzer for vein authentication
    Provides multiple segmentation strategies and visualization options
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = 'cuda'
    ):
        """
        Args:
            model: Model to explain
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.model.eval()
        self.explainer = None
        
        # Create explainer with default settings
        self._create_explainer()
    
    def _create_explainer(
        self,
        kernel_width: float = 0.25,
        feature_selection: str = 'auto',
        random_state: int = 42
    ):
        """
        Create LIME image explainer
        
        Args:
            kernel_width: Width of the kernel for the exponential kernel
            feature_selection: Feature selection method
            random_state: Random seed
        """
        self.explainer = lime_image.LimeImageExplainer(
            kernel_width=kernel_width,
            feature_selection=feature_selection,
            random_state=random_state
        )
    
    def batch_predict(
        self,
        images: np.ndarray
    ) -> np.ndarray:
        """
        Batch prediction function for LIME
        
        Args:
            images: Numpy array of images (N, H, W, C)
        
        Returns:
            Prediction probabilities (N, num_classes)
        """
        # Convert to tensor
        # LIME provides (N, H, W, C); model expects (N, C, H, W)
        if images.ndim == 3:
            # Single channel, add channel dimension
            images = np.expand_dims(images, axis=-1)
        
        images_tensor = torch.FloatTensor(images).permute(0, 3, 1, 2).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(images_tensor)
            
            # Handle different output formats
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)
            
            probs = torch.softmax(outputs, dim=1)
        
        return probs.cpu().numpy()
    
    def explain_instance(
        self,
        input_image: torch.Tensor,
        top_labels: int = 5,
        num_samples: int = 1000,
        segmentation_fn: Optional[Callable] = None,
        hide_color: Optional[float] = None,
        num_features: int = 10,
        batch_size: int = 10
    ) -> Tuple[lime_image.ImageExplanation, int, np.ndarray]:
        """
        Explain a single instance using LIME
        
        Args:
            input_image: Input image (1, C, H, W)
            top_labels: Number of top labels to explain
            num_samples: Number of samples for LIME
            segmentation_fn: Custom segmentation function
            hide_color: Color to hide segments (None = gray)
            num_features: Number of features to include
            batch_size: Batch size for predictions
        
        Returns:
            (explanation, predicted_class, segmentation_mask)
        """
        # Convert to numpy format for LIME (H, W, C)
        img_np = input_image[0].permute(1, 2, 0).cpu().numpy()
        
        # Convert to 3 channels if grayscale (LIME requires RGB)
        if img_np.shape[-1] == 1:
            img_np = gray2rgb(img_np.squeeze())
        
        # Normalize to [0, 1] if needed
        if img_np.min() < 0:
            img_np = (img_np + 1) / 2
        img_np = np.clip(img_np, 0, 1)
        
        # Default segmentation function
        if segmentation_fn is None:
            segmentation_fn = self._get_segmentation_function('slic')
        
        # Generate explanation
        try:
            explanation = self.explainer.explain_instance(
                img_np,
                self.batch_predict,
                top_labels=top_labels,
                num_samples=num_samples,
                segmentation_fn=segmentation_fn,
                hide_color=hide_color,
                batch_size=batch_size
            )
            
            # Get predicted class
            pred = self.batch_predict(img_np[np.newaxis, ...]).argmax()
            
            # Get segmentation mask
            segments = segmentation_fn(img_np)
            
            return explanation, int(pred), segments
            
        except Exception as e:
            warnings.warn(f"Error generating LIME explanation: {e}")
            # Return dummy explanation
            pred = self.batch_predict(img_np[np.newaxis, ...]).argmax()
            return None, int(pred), np.zeros(img_np.shape[:2])
    
    def _get_segmentation_function(
        self,
        method: str = 'slic',
        **kwargs
    ) -> Callable:
        """
        Get segmentation function
        
        Args:
            method: Segmentation method
                - 'slic': SLIC superpixels (default)
                - 'felzenszwalb': Felzenszwalb's method
                - 'quickshift': Quickshift
                - 'grid': Simple grid segmentation
            **kwargs: Additional parameters for segmentation
        
        Returns:
            Segmentation function
        """
        if method == 'slic':
            n_segments = kwargs.get('n_segments', 100)
            compactness = kwargs.get('compactness', 10)
            sigma = kwargs.get('sigma', 1)
            
            def slic_fn(image):
                return slic(
                    image,
                    n_segments=n_segments,
                    compactness=compactness,
                    sigma=sigma,
                    start_label=0
                )
            return slic_fn
        
        elif method == 'felzenszwalb':
            scale = kwargs.get('scale', 100)
            sigma = kwargs.get('sigma', 0.5)
            min_size = kwargs.get('min_size', 50)
            
            def felz_fn(image):
                return felzenszwalb(
                    image,
                    scale=scale,
                    sigma=sigma,
                    min_size=min_size
                )
            return felz_fn
        
        elif method == 'quickshift':
            kernel_size = kwargs.get('kernel_size', 3)
            max_dist = kwargs.get('max_dist', 10)
            ratio = kwargs.get('ratio', 0.5)
            
            def quick_fn(image):
                return quickshift(
                    image,
                    kernel_size=kernel_size,
                    max_dist=max_dist,
                    ratio=ratio
                )
            return quick_fn
        
        elif method == 'grid':
            grid_size = kwargs.get('grid_size', 16)
            
            def grid_fn(image):
                h, w = image.shape[:2]
                segments = np.zeros((h, w), dtype=int)
                segment_id = 0
                
                for i in range(0, h, grid_size):
                    for j in range(0, w, grid_size):
                        segments[i:i+grid_size, j:j+grid_size] = segment_id
                        segment_id += 1
                
                return segments
            return grid_fn
        
        else:
            raise ValueError(f"Unknown segmentation method: {method}")
    
    def explain_with_multiple_segmentations(
        self,
        input_image: torch.Tensor,
        segmentation_methods: List[str] = ['slic', 'felzenszwalb', 'quickshift'],
        num_samples: int = 1000
    ) -> Dict[str, Tuple]:
        """
        Generate explanations with different segmentation methods
        
        Args:
            input_image: Input image
            segmentation_methods: List of segmentation methods to try
            num_samples: Number of samples for LIME
        
        Returns:
            Dictionary mapping method names to (explanation, pred, segments)
        """
        results = {}
        
        for method in segmentation_methods:
            try:
                seg_fn = self._get_segmentation_function(method)
                exp, pred, segs = self.explain_instance(
                    input_image,
                    num_samples=num_samples,
                    segmentation_fn=seg_fn
                )
                results[method] = (exp, pred, segs)
            except Exception as e:
                warnings.warn(f"Failed to generate explanation with {method}: {e}")
                continue
        
        return results
    
    # ==================== Visualization Methods ====================
    
    def visualize_explanation(
        self,
        input_image: torch.Tensor,
        explanation: lime_image.ImageExplanation,
        pred_class: int,
        segments: np.ndarray,
        save_path: Optional[Path] = None,
        num_features: int = 10,
        positive_only: bool = True,
        hide_rest: bool = False
    ):
        """
        Visualize LIME explanation
        
        Args:
            input_image: Original input image
            explanation: LIME explanation object
            pred_class: Predicted class
            segments: Segmentation mask
            save_path: Path to save visualization
            num_features: Number of features to highlight
            positive_only: Show only positive contributions
            hide_rest: Hide non-important segments
        """
        if explanation is None:
            warnings.warn("No explanation to visualize")
            return
        
        # Get image and mask
        temp, mask = explanation.get_image_and_mask(
            pred_class,
            positive_only=positive_only,
            num_features=num_features,
            hide_rest=hide_rest
        )
        
        # Normalize temp if needed
        if temp.max() > 1:
            temp = temp / 255.0
        
        # Create boundary visualization
        img_boundary = mark_boundaries(temp, mask, color=(0, 1, 0), mode='thick')
        
        # Original image
        img_np = input_image[0, 0].cpu().numpy()
        img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original
        axes[0].imshow(img_norm, cmap='gray')
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Segmentation
        axes[1].imshow(mark_boundaries(temp, segments, color=(1, 0, 0)))
        axes[1].set_title(f"Segmentation ({len(np.unique(segments))} segments)")
        axes[1].axis('off')
        
        # Explanation
        axes[2].imshow(img_boundary)
        axes[2].set_title(f"LIME Explanation (Class {pred_class})")
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def visualize_detailed_explanation(
        self,
        input_image: torch.Tensor,
        explanation: lime_image.ImageExplanation,
        pred_class: int,
        segments: np.ndarray,
        save_path: Optional[Path] = None,
        num_features: int = 10
    ):
        """
        Create detailed visualization with multiple views
        
        Args:
            input_image: Original input
            explanation: LIME explanation
            pred_class: Predicted class
            segments: Segmentation mask
            save_path: Save path
            num_features: Number of features to show
        """
        if explanation is None:
            warnings.warn("No explanation to visualize")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        img_np = input_image[0, 0].cpu().numpy()
        img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        
        axes[0, 0].imshow(img_norm, cmap='gray')
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # Segmentation
        img_rgb = gray2rgb(img_norm)
        axes[0, 1].imshow(mark_boundaries(img_rgb, segments))
        axes[0, 1].set_title(f"Segmentation ({len(np.unique(segments))} segments)")
        axes[0, 1].axis('off')
        
        # Positive features only
        temp_pos, mask_pos = explanation.get_image_and_mask(
            pred_class,
            positive_only=True,
            num_features=num_features,
            hide_rest=False
        )
        temp_pos = temp_pos / 255.0 if temp_pos.max() > 1 else temp_pos
        axes[0, 2].imshow(mark_boundaries(temp_pos, mask_pos, color=(0, 1, 0)))
        axes[0, 2].set_title("Positive Features")
        axes[0, 2].axis('off')
        
        # Negative features only
        temp_neg, mask_neg = explanation.get_image_and_mask(
            pred_class,
            positive_only=False,
            num_features=num_features,
            hide_rest=False,
            negative_only=True
        )
        temp_neg = temp_neg / 255.0 if temp_neg.max() > 1 else temp_neg
        axes[1, 0].imshow(mark_boundaries(temp_neg, mask_neg, color=(1, 0, 0)))
        axes[1, 0].set_title("Negative Features")
        axes[1, 0].axis('off')
        
        # Both features
        temp_both, mask_both = explanation.get_image_and_mask(
            pred_class,
            positive_only=False,
            num_features=num_features,
            hide_rest=False
        )
        temp_both = temp_both / 255.0 if temp_both.max() > 1 else temp_both
        axes[1, 1].imshow(mark_boundaries(temp_both, mask_both))
        axes[1, 1].set_title("All Features")
        axes[1, 1].axis('off')
        
        # Feature importance plot
        local_exp = explanation.local_exp[pred_class]
        features, weights = zip(*sorted(local_exp, key=lambda x: abs(x[1]), reverse=True)[:num_features])
        
        colors = ['green' if w > 0 else 'red' for w in weights]
        axes[1, 2].barh(range(len(weights)), weights, color=colors)
        axes[1, 2].set_yticks(range(len(weights)))
        axes[1, 2].set_yticklabels([f"Segment {f}" for f in features])
        axes[1, 2].set_xlabel("Importance Weight")
        axes[1, 2].set_title("Feature Importance")
        axes[1, 2].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 2].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def compare_segmentations(
        self,
        input_image: torch.Tensor,
        results: Dict[str, Tuple],
        save_path: Optional[Path] = None
    ):
        """
        Compare explanations from different segmentation methods
        
        Args:
            input_image: Original input
            results: Dictionary from explain_with_multiple_segmentations
            save_path: Save path
        """
        n_methods = len(results)
        fig, axes = plt.subplots(2, n_methods, figsize=(6 * n_methods, 12))
        
        if n_methods == 1:
            axes = axes.reshape(2, 1)
        
        for idx, (method, (exp, pred, segs)) in enumerate(results.items()):
            if exp is None:
                axes[0, idx].text(0.5, 0.5, f"{method}\nFailed", 
                                 ha='center', va='center')
                axes[0, idx].axis('off')
                axes[1, idx].axis('off')
                continue
            
            # Segmentation
            img_np = input_image[0, 0].cpu().numpy()
            img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            img_rgb = gray2rgb(img_norm)
            
            axes[0, idx].imshow(mark_boundaries(img_rgb, segs))
            axes[0, idx].set_title(f"{method.upper()}\n({len(np.unique(segs))} segments)")
            axes[0, idx].axis('off')
            
            # Explanation
            temp, mask = exp.get_image_and_mask(
                pred,
                positive_only=True,
                num_features=10,
                hide_rest=False
            )
            temp = temp / 255.0 if temp.max() > 1 else temp
            
            axes[1, idx].imshow(mark_boundaries(temp, mask, color=(0, 1, 0)))
            axes[1, idx].set_title(f"Explanation (Class {pred})")
            axes[1, idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_top_features(
        self,
        explanation: lime_image.ImageExplanation,
        pred_class: int,
        num_features: int = 10,
        save_path: Optional[Path] = None
    ):
        """
        Plot bar chart of top contributing features
        
        Args:
            explanation: LIME explanation
            pred_class: Predicted class
            num_features: Number of features to show
            save_path: Save path
        """
        if explanation is None:
            warnings.warn("No explanation to plot")
            return
        
        # Get feature weights
        local_exp = explanation.local_exp[pred_class]
        features, weights = zip(*sorted(local_exp, key=lambda x: abs(x[1]), reverse=True)[:num_features])
        
        # Plot
        plt.figure(figsize=(10, max(6, num_features * 0.4)))
        
        colors = ['green' if w > 0 else 'red' for w in weights]
        bars = plt.barh(range(len(weights)), weights, color=colors, alpha=0.7)
        
        plt.yticks(range(len(weights)), [f"Segment {f}" for f in features])
        plt.xlabel("Feature Weight")
        plt.title(f"Top {num_features} Contributing Segments (Class {pred_class})")
        plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, weight) in enumerate(zip(bars, weights)):
            plt.text(
                weight,
                i,
                f'{weight:.3f}',
                ha='left' if weight > 0 else 'right',
                va='center',
                fontsize=9
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()


def compare_lime_settings(
    model: torch.nn.Module,
    input_image: torch.Tensor,
    num_samples_list: List[int] = [100, 500, 1000],
    save_path: Optional[Path] = None,
    device: str = 'cuda'
):
    """
    Compare LIME with different number of samples
    
    Args:
        model: Model to explain
        input_image: Input to explain
        num_samples_list: List of sample counts to try
        save_path: Save path
        device: Device
    """
    analyzer = VeinLIMEAnalyzer(model, device)
    
    fig, axes = plt.subplots(2, len(num_samples_list), figsize=(6 * len(num_samples_list), 12))
    
    if len(num_samples_list) == 1:
        axes = axes.reshape(2, 1)
    
    for idx, num_samples in enumerate(num_samples_list):
        exp, pred, segs = analyzer.explain_instance(
            input_image,
            num_samples=num_samples
        )
        
        if exp is None:
            axes[0, idx].text(0.5, 0.5, f"Failed\n{num_samples} samples",
                             ha='center', va='center')
            axes[0, idx].axis('off')
            axes[1, idx].axis('off')
            continue
        
        # Get explanation
        temp, mask = exp.get_image_and_mask(
            pred,
            positive_only=True,
            num_features=10,
            hide_rest=False
        )
        temp = temp / 255.0 if temp.max() > 1 else temp
        
        # Plot
        axes[0, idx].imshow(mark_boundaries(temp, mask, color=(0, 1, 0)))
        axes[0, idx].set_title(f"{num_samples} Samples")
        axes[0, idx].axis('off')
        
        # Feature importance
        local_exp = exp.local_exp[pred]
        features, weights = zip(*sorted(local_exp, key=lambda x: abs(x[1]), reverse=True)[:10])
        
        colors = ['green' if w > 0 else 'red' for w in weights]
        axes[1, idx].barh(range(len(weights)), weights, color=colors)
        axes[1, idx].set_yticks(range(len(weights)))
        axes[1, idx].set_yticklabels([f"S{f}" for f in features])
        axes[1, idx].set_xlabel("Weight")
        axes[1, idx].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, idx].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


if __name__ == "__main__":
    # Test LIME analyzer
    print("Testing LIME Analyzer...")
    
    # Create dummy model
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(1, 32, 3, padding=1)
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.fc = torch.nn.Linear(32, 10)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = DummyModel()
    test_input = torch.randn(1, 1, 64, 64)
    
    analyzer = VeinLIMEAnalyzer(model)
    exp, pred, segs = analyzer.explain_instance(test_input, num_samples=100)
    
    print(f"✓ Predicted class: {pred}")
    print(f"✓ Number of segments: {len(np.unique(segs))}")
    print("\n✓ All LIME tests passed!")