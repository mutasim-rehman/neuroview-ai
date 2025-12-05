"""
Training loop for brain tumor classification model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Dict, Optional
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from utils.helpers import get_device, save_checkpoint


class ClassificationTrainer:
    """Trainer class for brain tumor classification model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        num_classes: int = 4,
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
            num_classes: Number of classes
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
        self.num_classes = num_classes
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
        
        # Loss function - Cross Entropy for classification
        self.criterion = nn.CrossEntropyLoss()
        
        # TensorBoard writer
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            preds = torch.argmax(logits, dim=1)
            correct = (preds == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)
            
            # Collect for metrics
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total_loss += loss.item()
            
            # Log progress
            if (batch_idx + 1) % self.log_every_n_batches == 0:
                current_acc = total_correct / total_samples
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{current_acc:.4f}"
                })
                
                # Log to TensorBoard
                global_step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
                self.writer.add_scalar('Train/BatchAcc', current_acc, global_step)
        
        # Calculate epoch metrics
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = total_correct / total_samples
        
        # Calculate per-class metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
        avg_precision = np.mean(precision)
        avg_recall = np.mean(recall)
        avg_f1 = np.mean(f1)
        
        metrics = {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1
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
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")
            
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                
                # Calculate accuracy
                preds = torch.argmax(logits, dim=1)
                correct = (preds == labels).sum().item()
                total_correct += correct
                total_samples += labels.size(0)
                
                # Collect for metrics
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                total_loss += loss.item()
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{total_correct/total_samples:.4f}"
                })
        
        # Calculate epoch metrics
        epoch_loss = total_loss / len(self.val_loader)
        epoch_acc = total_correct / total_samples
        
        # Calculate per-class metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
        avg_precision = np.mean(precision)
        avg_recall = np.mean(recall)
        avg_f1 = np.mean(f1)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        metrics = {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1,
            'confusion_matrix': cm,
            'per_class_precision': precision.tolist(),
            'per_class_recall': recall.tolist(),
            'per_class_f1': f1.tolist()
        }
        
        return metrics
    
    def train(self, num_epochs: int, save_every_n_epochs: int = 5):
        """
        Train model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_every_n_epochs: Save checkpoint every N epochs
        """
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['loss'])
            self.train_accs.append(train_metrics['accuracy'])
            
            # Validate
            val_metrics = self.validate()
            self.val_losses.append(val_metrics['loss'])
            self.val_accs.append(val_metrics['accuracy'])
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log epoch metrics
            print(f"\nEpoch {epoch + 1}/{num_epochs}:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
                  f"F1: {train_metrics['f1']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            # Log to TensorBoard
            self.writer.add_scalar('Train/EpochLoss', train_metrics['loss'], epoch)
            self.writer.add_scalar('Train/EpochAcc', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Val/EpochLoss', val_metrics['loss'], epoch)
            self.writer.add_scalar('Val/EpochAcc', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('LearningRate', current_lr, epoch)
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_val_acc = val_metrics['accuracy']
                best_model_path = os.path.join(self.checkpoint_dir, "best_model.pth")
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics['loss'],
                    best_model_path
                )
                print(f"  ✓ New best model saved! (Loss: {self.best_val_loss:.4f}, Acc: {self.best_val_acc:.4f})")
            
            # Save checkpoint periodically
            if (epoch + 1) % save_every_n_epochs == 0:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f"checkpoint_epoch_{epoch + 1}.pth"
                )
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics['loss'],
                    checkpoint_path
                )
                print(f"  ✓ Checkpoint saved: {checkpoint_path}")
        
        self.writer.close()
        print(f"\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")

