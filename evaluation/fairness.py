"""
Fairness Analysis for Dorsal Hand Vein Recognition
Evaluates bias across demographic attributes like hand side
"""
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from config import config

class VeinFairnessAnalyzer:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def evaluate_fairness(self, data_loader, device='cuda'):
        """Analyzes model performance across different groups (L vs R)"""
        all_preds = []
        all_labels = []
        all_groups = []

        with torch.no_grad():
            for images, metadata in data_loader:
                images = images.to(device)
                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(metadata['person_id'])
                all_groups.extend(metadata['hand_side'])

        df = pd.DataFrame({
            'pred': all_preds,
            'label': all_labels,
            'group': all_groups
        })
        
        results = {}
        for group in df['group'].unique():
            group_df = df[df['group'] == group]
            accuracy = (group_df['pred'] == group_df['label']).mean()
            results[group] = {
                'accuracy': accuracy,
                'sample_size': len(group_df)
            }
            
        # Calculate Demographic Parity Difference
        # Ratio of positive predictions across groups
        group_stats = df.groupby('group')['pred'].value_counts(normalize=True)
        
        return results, self._calculate_fairness_metrics(df)

    def _calculate_fairness_metrics(self, df):
        """Calculates standard fairness metrics: Demographic Parity & Equal Opportunity"""
        metrics = {}
        groups = df['group'].unique()
        
        # Example for Binary Authentication scenario
        # (Simplified for multi-class person identification)
        accuracies = []
        for g in groups:
            g_df = df[df['group'] == g]
            accuracies.append((g_df['pred'] == g_df['label']).mean())
            
        metrics['max_accuracy_gap'] = max(accuracies) - min(accuracies)
        metrics['disparate_impact_ratio'] = min(accuracies) / max(accuracies)
        
        return metrics