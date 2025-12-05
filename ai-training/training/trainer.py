"""
Training loop and optimization for brain scan model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Dict, Optional
import os

from models.brain_model import BrainScanModel
from utils.helpers import get_device, save_checkpoint, calculate_metrics
import numpy as np


class Trainer:
    """Trainer class for brain scan model."""
    
    def __init__(
        self,
        model: BrainScanModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        log_dir: str = "./logs",
        checkpoint_dir: str = "./checkpoints",
        log_every_n_batches: int = 10
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            log_dir: Directory for logs
            checkpoint_dir: Directory for checkpoints
            log_every_n_batches: Log frequency
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.log_every_n_batches = log_every_n_batches
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        # Some older PyTorch versions don't support the `verbose` kwarg, so we
        # try to enable it and gracefully fall back if it's not accepted.
        try:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True,
            )
        except TypeError:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
            )
        
        # Loss function - combination of reconstruction and feature losses
        self.reconstruction_loss = nn.MSELoss()
        self.feature_loss = nn.MSELoss()
        
        # TensorBoard writer
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        
        for batch_idx, (volumes, metadata) in enumerate(pbar):
            volumes = volumes.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            reconstructed, encoded, features = self.model(volumes)
            
            # Reconstruction loss (main task)
            recon_loss = self.reconstruction_loss(reconstructed, volumes)
            
            # Feature consistency loss (encourage stable features)
            feature_loss = self.feature_loss(
                encoded.mean(dim=0).expand_as(encoded),
                encoded
            ) * 0.1  # Weight this loss lower
            
            # Total loss
            loss = recon_loss + feature_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recon': f"{recon_loss.item():.4f}"
            })
            
            # Log to tensorboard
            if batch_idx % self.log_every_n_batches == 0:
                global_step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
                self.writer.add_scalar('Train/BatchReconLoss', recon_loss.item(), global_step)
        
        # Average losses
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        
        metrics = {
            'loss': avg_loss,
            'reconstruction_loss': avg_recon_loss
        }
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """
        Validate on validation set.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_recon_loss = 0.0
        num_batches = 0
        
        all_reconstructions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")
            
            for volumes, metadata in pbar:
                volumes = volumes.to(self.device)
                
                # Forward pass
                reconstructed, encoded, features = self.model(volumes)
                
                # Losses
                recon_loss = self.reconstruction_loss(reconstructed, volumes)
                feature_loss = self.feature_loss(
                    encoded.mean(dim=0).expand_as(encoded),
                    encoded
                ) * 0.1
                
                loss = recon_loss + feature_loss
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                num_batches += 1
                
                # Collect for metrics
                all_reconstructions.append(reconstructed.cpu().numpy())
                all_targets.append(volumes.cpu().numpy())
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'recon': f"{recon_loss.item():.4f}"
                })
        
        # Average losses
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        
        # Calculate additional metrics
        all_reconstructions = np.concatenate(all_reconstructions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        additional_metrics = calculate_metrics(all_reconstructions, all_targets)
        
        metrics = {
            'loss': avg_loss,
            'reconstruction_loss': avg_recon_loss,
            **additional_metrics
        }
        
        return metrics
    
    def train(self, num_epochs: int, save_every_n_epochs: int = 5):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            save_every_n_epochs: Save checkpoint every N epochs
        """
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}\n")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['loss'])
            
            # Validate
            val_metrics = self.validate()
            val_loss = val_metrics['loss']
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Log to tensorboard
            self.writer.add_scalar('Train/EpochLoss', train_metrics['loss'], epoch)
            self.writer.add_scalar('Val/EpochLoss', val_loss, epoch)
            self.writer.add_scalar('Val/MSE', val_metrics['mse'], epoch)
            self.writer.add_scalar('Val/MAE', val_metrics['mae'], epoch)
            self.writer.add_scalar('LearningRate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print summary
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            print(f"  Val MSE: {val_metrics['mse']:.6f}")
            print(f"  Val MAE: {val_metrics['mae']:.6f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print(f"  ✓ New best validation loss: {val_loss:.6f}")
            
            if (epoch + 1) % save_every_n_epochs == 0 or is_best:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f"checkpoint_epoch_{epoch + 1}.pth"
                )
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_loss,
                    checkpoint_path,
                    best=is_best
                )
                
                if is_best:
                    best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        epoch,
                        val_loss,
                        best_path,
                        best=True
                    )
        
        self.writer.close()
        print(f"\nTraining completed! Best validation loss: {self.best_val_loss:.6f}")

