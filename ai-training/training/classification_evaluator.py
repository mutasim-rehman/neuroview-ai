"""
Evaluation module for brain tumor classification model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)


class ClassificationEvaluator:
    """Evaluator class for classification model testing."""
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        class_names: list = None
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model
            test_loader: Test data loader
            device: Device to evaluate on
            class_names: List of class names for reporting
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.class_names = class_names or [f"Class_{i}" for i in range(4)]
        self.model.eval()
    
    def evaluate(self) -> Dict:
        """
        Evaluate model on test set.
        
        Returns:
            Dictionary of evaluation metrics
        """
        print("\nEvaluating on test set...")
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Testing")
            
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(images)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                # Collect predictions
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
        
        # Average metrics
        avg_precision = np.mean(precision)
        avg_recall = np.mean(recall)
        avg_f1 = np.mean(f1)
        
        # Weighted metrics
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        # Macro metrics
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(avg_precision),
            'recall': float(avg_recall),
            'f1': float(avg_f1),
            'weighted_precision': float(weighted_precision),
            'weighted_recall': float(weighted_recall),
            'weighted_f1': float(weighted_f1),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'macro_f1': float(macro_f1),
            'confusion_matrix': cm.tolist(),
            'per_class_precision': precision.tolist(),
            'per_class_recall': recall.tolist(),
            'per_class_f1': f1.tolist(),
            'per_class_support': support.tolist(),
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }
        
        return metrics
    
    def print_results(self, metrics: Dict):
        """
        Print evaluation results in a formatted way.
        
        Args:
            metrics: Dictionary of evaluation metrics
        """
        print("\n" + "="*60)
        print("TEST SET EVALUATION RESULTS")
        print("="*60)
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f} (avg)")
        print(f"  Recall:    {metrics['recall']:.4f} (avg)")
        print(f"  F1-Score:  {metrics['f1']:.4f} (avg)")
        
        print(f"\nWeighted Metrics:")
        print(f"  Precision: {metrics['weighted_precision']:.4f}")
        print(f"  Recall:    {metrics['weighted_recall']:.4f}")
        print(f"  F1-Score:  {metrics['weighted_f1']:.4f}")
        
        print(f"\nMacro Metrics:")
        print(f"  Precision: {metrics['macro_precision']:.4f}")
        print(f"  Recall:    {metrics['macro_recall']:.4f}")
        print(f"  F1-Score:  {metrics['macro_f1']:.4f}")
        
        print(f"\nPer-Class Metrics:")
        print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 70)
        
        for i, class_name in enumerate(self.class_names):
            precision = metrics['per_class_precision'][i]
            recall = metrics['per_class_recall'][i]
            f1 = metrics['per_class_f1'][i]
            support = metrics['per_class_support'][i]
            
            print(f"{class_name:<20} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<10}")
        
        print(f"\nConfusion Matrix:")
        print(f"Rows = True labels, Columns = Predicted labels")
        print(f"{'':<15}", end="")
        for class_name in self.class_names:
            print(f"{class_name:<15}", end="")
        print()
        
        cm = np.array(metrics['confusion_matrix'])
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name:<15}", end="")
            for j in range(len(self.class_names)):
                print(f"{cm[i, j]:<15}", end="")
            print()
        
        print("="*60)
    
    def generate_classification_report(self, metrics: Dict) -> str:
        """
        Generate detailed classification report.
        
        Args:
            metrics: Dictionary of evaluation metrics
            
        Returns:
            Classification report string
        """
        report = classification_report(
            metrics['labels'],
            metrics['predictions'],
            target_names=self.class_names,
            output_dict=False
        )
        return report

