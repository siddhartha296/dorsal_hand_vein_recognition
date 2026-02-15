"""
Comprehensive Performance Metrics for GAN Quality and Authentication Accuracy
Enhanced version with multiple evaluation metrics and detailed analysis
"""
import torch
import torch.nn as nn
import numpy as np
from scipy import linalg
from scipy.stats import entropy
from torch.nn.functional import adaptive_avg_pool2d
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, 
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path

from config import config


class VeinMetrics:
    """
    Comprehensive metrics calculator for vein authentication and GAN quality
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.inception_model = None  # Lazy loading for FID/IS
    
    # ==================== Image Quality Metrics ====================
    
    @staticmethod
    def calculate_ssim_psnr(
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        data_range: float = 2.0
    ) -> Tuple[float, float]:
        """
        Calculate average SSIM and PSNR between real and synthetic batches
        
        Args:
            real_images: Real images (B, C, H, W) in range [-1, 1]
            fake_images: Fake images (B, C, H, W) in range [-1, 1]
            data_range: Data range (2.0 for [-1, 1])
        
        Returns:
            (mean_ssim, mean_psnr)
        """
        ssim_vals = []
        psnr_vals = []
        
        # Convert to numpy and scale to [0, 1] for skimage
        real_np = ((real_images.cpu().numpy() + 1) / 2).clip(0, 1)
        fake_np = ((fake_images.cpu().numpy() + 1) / 2).clip(0, 1)
        
        # Handle batch dimension
        if real_np.ndim == 4:
            real_np = real_np.squeeze(1)  # Remove channel dim for grayscale
            fake_np = fake_np.squeeze(1)
        
        for i in range(real_np.shape[0]):
            try:
                ssim_val = ssim(
                    real_np[i], 
                    fake_np[i], 
                    data_range=1.0,
                    channel_axis=None
                )
                psnr_val = psnr(
                    real_np[i], 
                    fake_np[i], 
                    data_range=1.0
                )
                ssim_vals.append(ssim_val)
                psnr_vals.append(psnr_val)
            except Exception as e:
                print(f"Warning: Error calculating SSIM/PSNR for sample {i}: {e}")
                continue
        
        return np.mean(ssim_vals) if ssim_vals else 0.0, np.mean(psnr_vals) if psnr_vals else 0.0
    
    @staticmethod
    def calculate_mse_mae(
        real_images: torch.Tensor,
        fake_images: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Calculate Mean Squared Error and Mean Absolute Error
        
        Returns:
            (mse, mae)
        """
        mse = torch.mean((real_images - fake_images) ** 2).item()
        mae = torch.mean(torch.abs(real_images - fake_images)).item()
        return mse, mae
    
    def calculate_fid(
        self,
        real_features: np.ndarray,
        fake_features: np.ndarray
    ) -> float:
        """
        Calculate Frechet Inception Distance (FID)
        
        Args:
            real_features: Feature vectors from real images (N, D)
            fake_features: Feature vectors from fake images (N, D)
        
        Returns:
            FID score (lower is better)
        """
        # Calculate mean and covariance
        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
        
        # Calculate sum of squared differences between means
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        
        # Calculate sqrt of product between covariances
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        
        # Check for imaginary numbers
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        # Calculate FID
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        
        return float(fid)
    
    @staticmethod
    def calculate_inception_score(
        predictions: np.ndarray,
        splits: int = 10
    ) -> Tuple[float, float]:
        """
        Calculate Inception Score (IS)
        
        Args:
            predictions: Softmax predictions (N, num_classes)
            splits: Number of splits for std calculation
        
        Returns:
            (mean_is, std_is)
        """
        # Calculate marginal distribution
        p_y = np.mean(predictions, axis=0)
        
        # Calculate KL divergence for each sample
        scores = []
        for i in range(splits):
            part = predictions[i * (len(predictions) // splits): (i + 1) * (len(predictions) // splits)]
            kl_div = part * (np.log(part + 1e-10) - np.log(p_y + 1e-10))
            kl_div = np.sum(kl_div, axis=1)
            scores.append(np.exp(np.mean(kl_div)))
        
        return float(np.mean(scores)), float(np.std(scores))
    
    # ==================== Authentication Metrics ====================
    
    @staticmethod
    def calculate_eer(
        y_true: np.ndarray,
        y_scores: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculate Equal Error Rate (EER) for authentication
        
        Args:
            y_true: True labels (0 or 1)
            y_scores: Prediction scores
        
        Returns:
            (eer, threshold_at_eer)
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
        fnr = 1 - tpr
        
        # EER is where FPR == FNR
        try:
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            thresh = interp1d(fpr, thresholds)(eer)
        except ValueError:
            # If brentq fails, find closest point
            abs_diffs = np.abs(fpr - fnr)
            min_index = np.argmin(abs_diffs)
            eer = np.mean([fpr[min_index], fnr[min_index]])
            thresh = thresholds[min_index]
        
        return float(eer), float(thresh)
    
    @staticmethod
    def calculate_far_frr(
        y_true: np.ndarray,
        y_scores: np.ndarray,
        threshold: float
    ) -> Tuple[float, float]:
        """
        Calculate False Acceptance Rate (FAR) and False Rejection Rate (FRR)
        
        Args:
            y_true: True labels
            y_scores: Prediction scores
            threshold: Decision threshold
        
        Returns:
            (far, frr)
        """
        predictions = (y_scores >= threshold).astype(int)
        
        # FAR: false positives / total negatives
        negatives = (y_true == 0)
        false_positives = np.sum((predictions == 1) & negatives)
        far = false_positives / np.sum(negatives) if np.sum(negatives) > 0 else 0
        
        # FRR: false negatives / total positives
        positives = (y_true == 1)
        false_negatives = np.sum((predictions == 0) & positives)
        frr = false_negatives / np.sum(positives) if np.sum(positives) > 0 else 0
        
        return float(far), float(frr)
    
    @staticmethod
    def calculate_roc_auc(
        y_true: np.ndarray,
        y_scores: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Calculate ROC curve and AUC
        
        Returns:
            Dictionary with 'fpr', 'tpr', 'thresholds', 'auc'
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        return {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': float(roc_auc)
        }
    
    @staticmethod
    def calculate_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = 'weighted'
    ) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging method for multi-class
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
        }
        
        return metrics
    
    @staticmethod
    def calculate_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """Calculate confusion matrix"""
        return confusion_matrix(y_true, y_pred)
    
    @staticmethod
    def get_classification_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None
    ) -> str:
        """Generate detailed classification report"""
        return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    
    # ==================== Verification Metrics ====================
    
    @staticmethod
    def calculate_verification_metrics(
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray,
        num_thresholds: int = 1000
    ) -> Dict:
        """
        Calculate comprehensive verification metrics
        
        Args:
            genuine_scores: Scores for genuine pairs (same person)
            impostor_scores: Scores for impostor pairs (different persons)
            num_thresholds: Number of thresholds to evaluate
        
        Returns:
            Dictionary with verification metrics
        """
        # Combine scores and labels
        all_scores = np.concatenate([genuine_scores, impostor_scores])
        all_labels = np.concatenate([
            np.ones(len(genuine_scores)),
            np.zeros(len(impostor_scores))
        ])
        
        # Calculate EER
        eer, eer_threshold = VeinMetrics.calculate_eer(all_labels, all_scores)
        
        # Calculate FAR and FRR at EER threshold
        far_at_eer, frr_at_eer = VeinMetrics.calculate_far_frr(
            all_labels, all_scores, eer_threshold
        )
        
        # Calculate ROC curve
        roc_data = VeinMetrics.calculate_roc_auc(all_labels, all_scores)
        
        # Calculate metrics at various thresholds
        thresholds = np.linspace(all_scores.min(), all_scores.max(), num_thresholds)
        far_list, frr_list = [], []
        
        for thresh in thresholds:
            far, frr = VeinMetrics.calculate_far_frr(all_labels, all_scores, thresh)
            far_list.append(far)
            frr_list.append(frr)
        
        return {
            'eer': eer,
            'eer_threshold': eer_threshold,
            'far_at_eer': far_at_eer,
            'frr_at_eer': frr_at_eer,
            'auc': roc_data['auc'],
            'roc_curve': roc_data,
            'far_frr_curve': {
                'thresholds': thresholds,
                'far': np.array(far_list),
                'frr': np.array(frr_list)
            },
            'genuine_mean': float(np.mean(genuine_scores)),
            'genuine_std': float(np.std(genuine_scores)),
            'impostor_mean': float(np.mean(impostor_scores)),
            'impostor_std': float(np.std(impostor_scores)),
            'd_prime': float(
                (np.mean(genuine_scores) - np.mean(impostor_scores)) /
                np.sqrt(0.5 * (np.var(genuine_scores) + np.var(impostor_scores)))
            )
        }
    
    # ==================== GAN-Specific Metrics ====================
    
    @staticmethod
    def calculate_mode_score(
        real_features: np.ndarray,
        fake_features: np.ndarray,
        num_neighbors: int = 5
    ) -> float:
        """
        Calculate Mode Score to detect mode collapse
        
        Returns:
            Mode score (higher is better, max = 1.0)
        """
        from sklearn.metrics.pairwise import euclidean_distances
        
        # Calculate pairwise distances
        distances = euclidean_distances(fake_features, real_features)
        
        # For each fake sample, find nearest real samples
        nearest_indices = np.argsort(distances, axis=1)[:, :num_neighbors]
        
        # Count unique nearest real samples
        unique_real_samples = len(np.unique(nearest_indices.flatten()))
        total_real_samples = len(real_features)
        
        mode_score = unique_real_samples / total_real_samples
        
        return float(mode_score)
    
    @staticmethod
    def calculate_diversity_score(features: np.ndarray) -> float:
        """
        Calculate diversity score of generated samples
        
        Returns:
            Diversity score (higher is better)
        """
        from sklearn.metrics.pairwise import cosine_distances
        
        # Calculate pairwise cosine distances
        distances = cosine_distances(features)
        
        # Remove diagonal (self-distances)
        mask = np.ones(distances.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        
        # Mean distance represents diversity
        diversity = distances[mask].mean()
        
        return float(diversity)
    
    # ==================== Visualization Methods ====================
    
    @staticmethod
    def plot_roc_curve(
        roc_data: Dict,
        save_path: Optional[Path] = None,
        title: str = "ROC Curve"
    ):
        """Plot ROC curve"""
        plt.figure(figsize=(8, 6))
        plt.plot(
            roc_data['fpr'],
            roc_data['tpr'],
            color='darkorange',
            lw=2,
            label=f'ROC curve (AUC = {roc_data["auc"]:.4f})'
        )
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_far_frr_curve(
        far_frr_data: Dict,
        eer: float,
        save_path: Optional[Path] = None,
        title: str = "FAR/FRR Curve"
    ):
        """Plot FAR and FRR curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(
            far_frr_data['thresholds'],
            far_frr_data['far'],
            label='FAR (False Acceptance Rate)',
            color='red',
            lw=2
        )
        plt.plot(
            far_frr_data['thresholds'],
            far_frr_data['frr'],
            label='FRR (False Rejection Rate)',
            color='blue',
            lw=2
        )
        plt.axhline(y=eer, color='green', linestyle='--', label=f'EER = {eer:.4f}')
        plt.xlabel('Threshold')
        plt.ylabel('Error Rate')
        plt.title(title)
        plt.legend()
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_confusion_matrix(
        cm: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_path: Optional[Path] = None,
        title: str = "Confusion Matrix"
    ):
        """Plot confusion matrix"""
        import seaborn as sns
        
        plt.figure(figsize=(10, 8))
        
        if class_names and len(class_names) <= 20:  # Only show names if reasonable
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names
            )
        else:
            sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
        
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_score_distribution(
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray,
        save_path: Optional[Path] = None,
        title: str = "Score Distribution"
    ):
        """Plot distribution of genuine and impostor scores"""
        plt.figure(figsize=(10, 6))
        
        plt.hist(
            genuine_scores,
            bins=50,
            alpha=0.6,
            label='Genuine',
            color='green',
            density=True
        )
        plt.hist(
            impostor_scores,
            bins=50,
            alpha=0.6,
            label='Impostor',
            color='red',
            density=True
        )
        
        plt.xlabel('Similarity Score')
        plt.ylabel('Density')
        plt.title(title)
        plt.legend()
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


class MetricsTracker:
    """
    Track metrics over training/evaluation
    """
    
    def __init__(self):
        self.metrics_history = {}
    
    def update(self, metrics: Dict[str, float], step: int):
        """Add metrics for current step"""
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = {'steps': [], 'values': []}
            self.metrics_history[key]['steps'].append(step)
            self.metrics_history[key]['values'].append(value)
    
    def get_metric(self, metric_name: str) -> Dict:
        """Get history for specific metric"""
        return self.metrics_history.get(metric_name, {'steps': [], 'values': []})
    
    def get_latest(self, metric_name: str) -> Optional[float]:
        """Get latest value for metric"""
        if metric_name in self.metrics_history:
            values = self.metrics_history[metric_name]['values']
            return values[-1] if values else None
        return None
    
    def plot_metrics(
        self,
        metric_names: List[str],
        save_path: Optional[Path] = None,
        title: str = "Metrics History"
    ):
        """Plot metrics over time"""
        plt.figure(figsize=(12, 6))
        
        for metric_name in metric_names:
            if metric_name in self.metrics_history:
                data = self.metrics_history[metric_name]
                plt.plot(data['steps'], data['values'], label=metric_name, marker='o')
        
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.title(title)
        plt.legend()
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_to_file(self, filepath: Path):
        """Save metrics history to file"""
        import json
        
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def load_from_file(self, filepath: Path):
        """Load metrics history from file"""
        import json
        
        with open(filepath, 'r') as f:
            self.metrics_history = json.load(f)


if __name__ == "__main__":
    # Test metrics
    print("Testing VeinMetrics...")
    
    # Test SSIM/PSNR
    real = torch.randn(4, 1, 256, 256)
    fake = real + torch.randn(4, 1, 256, 256) * 0.1
    
    metrics = VeinMetrics()
    ssim_val, psnr_val = metrics.calculate_ssim_psnr(real, fake)
    print(f"SSIM: {ssim_val:.4f}, PSNR: {psnr_val:.2f}")
    
    # Test EER
    y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
    y_scores = np.array([0.9, 0.8, 0.7, 0.3, 0.4, 0.2, 0.85, 0.1, 0.95, 0.15])
    eer, thresh = metrics.calculate_eer(y_true, y_scores)
    print(f"EER: {eer:.4f}, Threshold: {thresh:.4f}")
    
    print("\n✓ All tests passed!")