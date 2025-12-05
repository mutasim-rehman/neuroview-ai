"""
Evaluation module for testing the trained model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict
import numpy as np

from models.brain_model import BrainScanModel
from utils.helpers import calculate_metrics, get_device


class Evaluator:
    """Evaluator class for model testing."""
    
    def __init__(
        self,
        model: BrainScanModel,
        test_loader: DataLoader,
        device: torch.device
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model
            test_loader: Test data loader
            device: Device to evaluate on
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.model.eval()
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Returns:
            Dictionary of evaluation metrics
        """
        print("\nEvaluating on test set...")
        
        total_loss = 0.0
        total_recon_loss = 0.0
        num_batches = 0
        
        all_reconstructions = []
        all_targets = []
        all_features = []
        
        reconstruction_loss = nn.MSELoss()
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Testing")
            
            for volumes, metadata in pbar:
                volumes = volumes.to(self.device)
                
                # Forward pass
                reconstructed, encoded, features = self.model(volumes)
                
                # Reconstruction loss
                recon_loss = reconstruction_loss(reconstructed, volumes)
                loss = recon_loss.item()
                
                total_loss += loss
                total_recon_loss += recon_loss.item()
                num_batches += 1
                
                # Collect for metrics
                all_reconstructions.append(reconstructed.cpu().numpy())
                all_targets.append(volumes.cpu().numpy())
                all_features.append(encoded.cpu().numpy())
                
                pbar.set_postfix({'loss': f"{loss:.4f}"})
        
        # Calculate metrics
        all_reconstructions = np.concatenate(all_reconstructions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        metrics = calculate_metrics(all_reconstructions, all_targets)
        metrics['loss'] = total_loss / num_batches
        metrics['reconstruction_loss'] = total_recon_loss / num_batches
        
        # Feature statistics
        all_features = np.concatenate(all_features, axis=0)
        metrics['feature_mean'] = float(np.mean(all_features))
        metrics['feature_std'] = float(np.std(all_features))
        
        return metrics
    
    def print_results(self, metrics: Dict[str, float]):
        """
        Print evaluation results in a formatted way.
        
        Args:
            metrics: Dictionary of metrics
        """
        print("\n" + "="*50)
        print("TEST SET EVALUATION RESULTS")
        print("="*50)
        print(f"Loss:              {metrics['loss']:.6f}")
        print(f"Reconstruction Loss: {metrics['reconstruction_loss']:.6f}")
        print(f"MSE:               {metrics['mse']:.6f}")
        print(f"MAE:               {metrics['mae']:.6f}")
        print(f"RMSE:              {metrics['rmse']:.6f}")
        print(f"NRMSE:             {metrics['nrmse']:.6f}")
        print(f"Feature Mean:      {metrics['feature_mean']:.6f}")
        print(f"Feature Std:       {metrics['feature_std']:.6f}")
        print("="*50 + "\n")

