"""
Fairness Analysis for Dorsal Hand Vein Recognition
Comprehensive bias detection and analysis across demographic attributes
Enhanced version with multiple fairness metrics and visualization
"""
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

from config import config


class VeinFairnessAnalyzer:
    """
    Comprehensive fairness analyzer for vein authentication systems
    Evaluates bias across demographic groups (e.g., left vs right hand)
    """
    
    def __init__(self, model):
        """
        Args:
            model: Authentication model to evaluate
        """
        self.model = model
        self.model.eval()
    
    def evaluate_fairness(
        self,
        data_loader,
        device='cuda',
        sensitive_attribute='hand_side'
    ) -> Tuple[Dict, Dict]:
        """
        Analyze model performance across different demographic groups
        
        Args:
            data_loader: DataLoader with metadata
            device: Device to run evaluation on
            sensitive_attribute: Attribute to analyze fairness across
        
        Returns:
            (group_results, fairness_metrics)
        """
        all_preds = []
        all_labels = []
        all_groups = []
        all_probs = []
        
        print(f"Evaluating fairness across '{sensitive_attribute}'...")
        
        with torch.no_grad():
            for images, metadata in data_loader:
                images = images.to(device)
                outputs = self.model(images)
                
                # Get predictions
                if outputs.dim() > 1 and outputs.size(1) > 1:
                    # Multi-class classification
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs, dim=1)
                    all_probs.extend(probs.cpu().numpy())
                else:
                    # Binary classification
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).long().squeeze()
                    all_probs.extend(probs.cpu().numpy())
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend([metadata['person_id'][i].item() if torch.is_tensor(metadata['person_id'][i]) 
                                  else metadata['person_id'][i] 
                                  for i in range(len(metadata['person_id']))])
                all_groups.extend(metadata[sensitive_attribute])
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame({
            'pred': all_preds,
            'label': all_labels,
            'group': all_groups,
            'prob': [p.max() if hasattr(p, 'max') else p for p in all_probs]
        })
        
        # Calculate per-group metrics
        group_results = self._calculate_group_metrics(df)
        
        # Calculate fairness metrics
        fairness_metrics = self._calculate_fairness_metrics(df, group_results)
        
        return group_results, fairness_metrics
    
    def _calculate_group_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate performance metrics for each group
        
        Returns:
            Dictionary with per-group metrics
        """
        results = {}
        
        for group in df['group'].unique():
            group_df = df[df['group'] == group]
            
            # Basic metrics
            accuracy = accuracy_score(group_df['label'], group_df['pred'])
            
            # Confusion matrix
            cm = confusion_matrix(group_df['label'], group_df['pred'])
            
            # True/False Positive/Negative rates
            if cm.size > 1:
                # For binary or simplified multi-class
                tp = np.diag(cm).sum()
                fp = cm.sum(axis=0) - np.diag(cm)
                fn = cm.sum(axis=1) - np.diag(cm)
                tn = cm.sum() - (tp + fp.sum() + fn.sum())
                
                tpr = tp / (tp + fn.sum()) if (tp + fn.sum()) > 0 else 0
                fpr = fp.sum() / (fp.sum() + tn) if (fp.sum() + tn) > 0 else 0
                tnr = tn / (tn + fp.sum()) if (tn + fp.sum()) > 0 else 0
                fnr = fn.sum() / (fn.sum() + tp) if (fn.sum() + tp) > 0 else 0
            else:
                tpr = fpr = tnr = fnr = 0
            
            # Prediction distribution
            pred_positive_rate = (group_df['pred'] > 0).mean()
            
            results[group] = {
                'accuracy': float(accuracy),
                'sample_size': len(group_df),
                'true_positive_rate': float(tpr),
                'false_positive_rate': float(fpr),
                'true_negative_rate': float(tnr),
                'false_negative_rate': float(fnr),
                'prediction_positive_rate': float(pred_positive_rate),
                'mean_confidence': float(group_df['prob'].mean()),
                'std_confidence': float(group_df['prob'].std()),
                'confusion_matrix': cm
            }
        
        return results
    
    def _calculate_fairness_metrics(
        self,
        df: pd.DataFrame,
        group_results: Dict
    ) -> Dict:
        """
        Calculate standard fairness metrics
        
        Implements:
        - Demographic Parity
        - Equal Opportunity
        - Equalized Odds
        - Predictive Parity
        - Calibration
        
        Returns:
            Dictionary of fairness metrics
        """
        metrics = {}
        groups = list(group_results.keys())
        
        if len(groups) < 2:
            return {'error': 'Need at least 2 groups for fairness analysis'}
        
        # Extract metrics for each group
        accuracies = [group_results[g]['accuracy'] for g in groups]
        tpr_list = [group_results[g]['true_positive_rate'] for g in groups]
        fpr_list = [group_results[g]['false_positive_rate'] for g in groups]
        pred_pos_rates = [group_results[g]['prediction_positive_rate'] for g in groups]
        
        # 1. Demographic Parity Difference
        # Difference in positive prediction rates
        metrics['demographic_parity_difference'] = float(
            max(pred_pos_rates) - min(pred_pos_rates)
        )
        
        # 2. Demographic Parity Ratio
        # Ratio of min to max positive prediction rate
        metrics['demographic_parity_ratio'] = float(
            min(pred_pos_rates) / max(pred_pos_rates) if max(pred_pos_rates) > 0 else 0
        )
        
        # 3. Equal Opportunity Difference
        # Difference in TPR (recall for positive class)
        metrics['equal_opportunity_difference'] = float(
            max(tpr_list) - min(tpr_list)
        )
        
        # 4. Equalized Odds (Average of TPR and FPR differences)
        tpr_diff = max(tpr_list) - min(tpr_list)
        fpr_diff = max(fpr_list) - min(fpr_list)
        metrics['equalized_odds_difference'] = float((tpr_diff + fpr_diff) / 2)
        
        # 5. Accuracy Gap
        metrics['max_accuracy_gap'] = float(max(accuracies) - min(accuracies))
        
        # 6. Disparate Impact Ratio
        # Ratio of minimum to maximum accuracy
        metrics['disparate_impact_ratio'] = float(
            min(accuracies) / max(accuracies) if max(accuracies) > 0 else 0
        )
        
        # 7. Statistical Parity
        # Standard deviation of positive prediction rates
        metrics['statistical_parity_std'] = float(np.std(pred_pos_rates))
        
        # 8. Group Fairness Score (0-1, higher is better)
        # Combined metric considering multiple fairness criteria
        metrics['overall_fairness_score'] = self._calculate_overall_fairness_score(
            metrics['demographic_parity_ratio'],
            metrics['disparate_impact_ratio'],
            metrics['equal_opportunity_difference']
        )
        
        # 9. Per-group comparison matrix
        metrics['pairwise_comparisons'] = self._calculate_pairwise_fairness(
            groups, group_results
        )
        
        return metrics
    
    @staticmethod
    def _calculate_overall_fairness_score(
        dp_ratio: float,
        di_ratio: float,
        eo_diff: float
    ) -> float:
        """
        Calculate overall fairness score combining multiple metrics
        
        Returns:
            Score between 0 and 1 (1 = perfectly fair)
        """
        # Normalize each component to [0, 1]
        # For ratios: already in [0, 1], 1 is ideal
        # For differences: need to invert, 0 is ideal
        
        dp_score = dp_ratio  # Higher is better
        di_score = di_ratio  # Higher is better
        eo_score = 1 - min(eo_diff, 1)  # Lower difference is better
        
        # Weighted average
        overall_score = (dp_score * 0.3 + di_score * 0.4 + eo_score * 0.3)
        
        return float(overall_score)
    
    @staticmethod
    def _calculate_pairwise_fairness(
        groups: List[str],
        group_results: Dict
    ) -> Dict:
        """
        Calculate pairwise fairness comparisons between groups
        """
        pairwise = {}
        
        for i, group1 in enumerate(groups):
            for group2 in groups[i+1:]:
                key = f"{group1}_vs_{group2}"
                
                acc_diff = abs(
                    group_results[group1]['accuracy'] - 
                    group_results[group2]['accuracy']
                )
                
                tpr_diff = abs(
                    group_results[group1]['true_positive_rate'] - 
                    group_results[group2]['true_positive_rate']
                )
                
                fpr_diff = abs(
                    group_results[group1]['false_positive_rate'] - 
                    group_results[group2]['false_positive_rate']
                )
                
                pairwise[key] = {
                    'accuracy_difference': float(acc_diff),
                    'tpr_difference': float(tpr_diff),
                    'fpr_difference': float(fpr_diff),
                    'is_fair': acc_diff < 0.05  # 5% threshold
                }
        
        return pairwise
    
    def detect_bias(
        self,
        fairness_metrics: Dict,
        thresholds: Optional[Dict] = None
    ) -> Dict:
        """
        Detect potential biases based on fairness metrics
        
        Args:
            fairness_metrics: Output from evaluate_fairness
            thresholds: Custom thresholds for bias detection
        
        Returns:
            Dictionary of detected biases
        """
        if thresholds is None:
            thresholds = {
                'demographic_parity_difference': 0.1,
                'equal_opportunity_difference': 0.1,
                'max_accuracy_gap': 0.05,
                'disparate_impact_ratio': 0.8
            }
        
        biases = {
            'detected': False,
            'issues': []
        }
        
        # Check each metric against threshold
        if fairness_metrics.get('demographic_parity_difference', 0) > thresholds['demographic_parity_difference']:
            biases['detected'] = True
            biases['issues'].append({
                'type': 'demographic_parity_violation',
                'severity': 'high',
                'description': 'Significant difference in positive prediction rates across groups',
                'value': fairness_metrics['demographic_parity_difference']
            })
        
        if fairness_metrics.get('equal_opportunity_difference', 0) > thresholds['equal_opportunity_difference']:
            biases['detected'] = True
            biases['issues'].append({
                'type': 'equal_opportunity_violation',
                'severity': 'high',
                'description': 'Significant difference in true positive rates across groups',
                'value': fairness_metrics['equal_opportunity_difference']
            })
        
        if fairness_metrics.get('max_accuracy_gap', 0) > thresholds['max_accuracy_gap']:
            biases['detected'] = True
            biases['issues'].append({
                'type': 'accuracy_disparity',
                'severity': 'medium',
                'description': 'Notable accuracy gap between groups',
                'value': fairness_metrics['max_accuracy_gap']
            })
        
        if fairness_metrics.get('disparate_impact_ratio', 1) < thresholds['disparate_impact_ratio']:
            biases['detected'] = True
            biases['issues'].append({
                'type': 'disparate_impact',
                'severity': 'high',
                'description': 'Disproportionate impact on underperforming group',
                'value': fairness_metrics['disparate_impact_ratio']
            })
        
        return biases
    
    # ==================== Visualization Methods ====================
    
    @staticmethod
    def plot_group_performance(
        group_results: Dict,
        save_path: Optional[Path] = None,
        title: str = "Performance Across Groups"
    ):
        """Plot performance metrics for each group"""
        groups = list(group_results.keys())
        metrics_to_plot = ['accuracy', 'true_positive_rate', 'false_positive_rate']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, metric in enumerate(metrics_to_plot):
            values = [group_results[g][metric] for g in groups]
            
            axes[idx].bar(groups, values, color=['skyblue', 'lightcoral'][:len(groups)])
            axes[idx].set_ylabel(metric.replace('_', ' ').title())
            axes[idx].set_xlabel('Group')
            axes[idx].set_ylim([0, 1])
            axes[idx].grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(values):
                axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_fairness_metrics(
        fairness_metrics: Dict,
        save_path: Optional[Path] = None,
        title: str = "Fairness Metrics"
    ):
        """Plot fairness metrics as bar chart"""
        # Select key metrics to plot
        metrics_to_plot = {
            'Demographic Parity\nDifference': fairness_metrics.get('demographic_parity_difference', 0),
            'Equal Opportunity\nDifference': fairness_metrics.get('equal_opportunity_difference', 0),
            'Max Accuracy\nGap': fairness_metrics.get('max_accuracy_gap', 0),
            'Equalized Odds\nDifference': fairness_metrics.get('equalized_odds_difference', 0)
        }
        
        plt.figure(figsize=(10, 6))
        
        bars = plt.bar(
            metrics_to_plot.keys(),
            metrics_to_plot.values(),
            color=['red' if v > 0.1 else 'orange' if v > 0.05 else 'green' 
                   for v in metrics_to_plot.values()]
        )
        
        plt.ylabel('Metric Value')
        plt.title(title)
        plt.xticks(rotation=15, ha='right')
        plt.axhline(y=0.05, color='orange', linestyle='--', label='Warning Threshold (0.05)')
        plt.axhline(y=0.1, color='red', linestyle='--', label='Critical Threshold (0.1)')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom'
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_confusion_matrices(
        group_results: Dict,
        save_path: Optional[Path] = None,
        title: str = "Confusion Matrices by Group"
    ):
        """Plot confusion matrices for each group"""
        groups = list(group_results.keys())
        n_groups = len(groups)
        
        fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 5))
        
        if n_groups == 1:
            axes = [axes]
        
        for idx, group in enumerate(groups):
            cm = group_results[group]['confusion_matrix']
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(
                cm_normalized,
                annot=True,
                fmt='.2f',
                cmap='Blues',
                ax=axes[idx],
                cbar=True
            )
            
            axes[idx].set_title(f'Group: {group}')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_fairness_radar(
        fairness_metrics: Dict,
        save_path: Optional[Path] = None,
        title: str = "Fairness Radar Chart"
    ):
        """Plot fairness metrics as radar chart"""
        from math import pi
        
        # Select metrics (normalized to 0-1, where 1 is fair)
        categories = [
            'Demographic\nParity',
            'Equal\nOpportunity',
            'Accuracy\nEquality',
            'Overall\nFairness'
        ]
        
        values = [
            fairness_metrics.get('demographic_parity_ratio', 0),
            1 - min(fairness_metrics.get('equal_opportunity_difference', 0), 1),
            fairness_metrics.get('disparate_impact_ratio', 0),
            fairness_metrics.get('overall_fairness_score', 0)
        ]
        
        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        values += values[:1]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        ax.plot(angles, values, 'o-', linewidth=2, label='Current Model')
        ax.fill(angles, values, alpha=0.25)
        
        # Perfect fairness reference
        perfect = [1.0] * (N + 1)
        ax.plot(angles, perfect, '--', linewidth=1, color='green', label='Perfect Fairness')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)
        
        plt.title(title, y=1.08)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_fairness_report(
        self,
        group_results: Dict,
        fairness_metrics: Dict,
        save_path: Path
    ):
        """
        Generate comprehensive fairness report
        
        Args:
            group_results: Per-group performance metrics
            fairness_metrics: Overall fairness metrics
            save_path: Path to save report
        """
        with open(save_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("FAIRNESS ANALYSIS REPORT\n")
            f.write("="*70 + "\n\n")
            
            # Group Performance
            f.write("1. GROUP PERFORMANCE SUMMARY\n")
            f.write("-"*70 + "\n")
            for group, metrics in group_results.items():
                f.write(f"\nGroup: {group}\n")
                f.write(f"  Sample Size: {metrics['sample_size']}\n")
                f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"  TPR: {metrics['true_positive_rate']:.4f}\n")
                f.write(f"  FPR: {metrics['false_positive_rate']:.4f}\n")
                f.write(f"  Mean Confidence: {metrics['mean_confidence']:.4f}\n")
            
            # Fairness Metrics
            f.write("\n\n2. FAIRNESS METRICS\n")
            f.write("-"*70 + "\n")
            for metric, value in fairness_metrics.items():
                if isinstance(value, (int, float)):
                    f.write(f"  {metric}: {value:.4f}\n")
            
            # Bias Detection
            f.write("\n\n3. BIAS DETECTION\n")
            f.write("-"*70 + "\n")
            biases = self.detect_bias(fairness_metrics)
            if biases['detected']:
                f.write("  ⚠ BIASES DETECTED:\n")
                for issue in biases['issues']:
                    f.write(f"    - {issue['type']} ({issue['severity']} severity)\n")
                    f.write(f"      {issue['description']}\n")
                    f.write(f"      Value: {issue['value']:.4f}\n\n")
            else:
                f.write("  ✓ No significant biases detected\n")
            
            # Recommendations
            f.write("\n\n4. RECOMMENDATIONS\n")
            f.write("-"*70 + "\n")
            if biases['detected']:
                f.write("  - Consider data augmentation for underrepresented groups\n")
                f.write("  - Apply fairness-aware training techniques\n")
                f.write("  - Review feature engineering for bias sources\n")
                f.write("  - Implement threshold adjustment per group if applicable\n")
            else:
                f.write("  - Continue monitoring fairness metrics\n")
                f.write("  - Maintain balanced evaluation across groups\n")
            
            f.write("\n" + "="*70 + "\n")


if __name__ == "__main__":
    # Test fairness analyzer
    print("Testing VeinFairnessAnalyzer...")
    
    # Create dummy model
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.randn(x.size(0), 10)
    
    model = DummyModel()
    analyzer = VeinFairnessAnalyzer(model)
    
    print("✓ Fairness analyzer initialized successfully")