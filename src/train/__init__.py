"""Training utilities for graph diffusion models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from typing import Dict, Any, Optional, Tuple, List
import logging
from tqdm import tqdm
import numpy as np
from torchmetrics import Accuracy, F1Score, AUROC
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassAUROC

from ..utils import EarlyStopping, save_checkpoint, load_checkpoint


class Trainer:
    """Trainer class for graph diffusion models."""
    
    def __init__(
        self,
        model: nn.Module,
        data,
        config: Dict[str, Any],
        device: torch.device,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize trainer.
        
        Args:
            model: PyTorch model
            data: PyTorch Geometric data object
            config: Training configuration
            device: Device to use for training
            logger: Logger instance
        """
        self.model = model.to(device)
        self.data = data.to(device)
        self.config = config
        self.device = device
        self.logger = logger or logging.getLogger("graph_diffusion")
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get("patience", 50),
            min_delta=config.get("min_delta", 0.0),
            restore_best_weights=True
        ) if config.get("early_stopping", True) else None
        
        # Setup metrics
        self._setup_metrics()
        
        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "test_loss": [],
            "train_acc": [],
            "val_acc": [],
            "test_acc": [],
            "train_f1": [],
            "val_f1": [],
            "test_f1": [],
            "lr": []
        }
        
        self.best_metrics = {}
    
    def _setup_optimizer(self):
        """Setup optimizer."""
        optimizer_type = self.config.get("optimizer", "adam").lower()
        
        if optimizer_type == "adam":
            return Adam(
                self.model.parameters(),
                lr=self.config.get("lr", 0.01),
                weight_decay=self.config.get("weight_decay", 5e-4)
            )
        elif optimizer_type == "adamw":
            return AdamW(
                self.model.parameters(),
                lr=self.config.get("lr", 0.01),
                weight_decay=self.config.get("weight_decay", 5e-4)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        scheduler_type = self.config.get("scheduler", "plateau").lower()
        
        if scheduler_type == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=10,
                verbose=True
            )
        elif scheduler_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get("epochs", 200)
            )
        else:
            return None
    
    def _setup_metrics(self):
        """Setup evaluation metrics."""
        num_classes = self.data.num_classes
        
        self.train_metrics = {
            "accuracy": MulticlassAccuracy(num_classes=num_classes, average="micro"),
            "f1": MulticlassF1Score(num_classes=num_classes, average="macro"),
            "auroc": MulticlassAUROC(num_classes=num_classes, average="macro")
        }
        
        self.val_metrics = {
            "accuracy": MulticlassAccuracy(num_classes=num_classes, average="micro"),
            "f1": MulticlassF1Score(num_classes=num_classes, average="macro"),
            "auroc": MulticlassAUROC(num_classes=num_classes, average="macro")
        }
        
        self.test_metrics = {
            "accuracy": MulticlassAccuracy(num_classes=num_classes, average="micro"),
            "f1": MulticlassF1Score(num_classes=num_classes, average="macro"),
            "auroc": MulticlassAUROC(num_classes=num_classes, average="macro")
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        out = self.model(self.data)
        loss = F.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Compute metrics
        pred = out[self.data.train_mask].argmax(dim=1)
        true = self.data.y[self.data.train_mask]
        
        metrics = {
            "loss": loss.item(),
            "accuracy": self.train_metrics["accuracy"](pred, true).item(),
            "f1": self.train_metrics["f1"](pred, true).item(),
            "auroc": self.train_metrics["auroc"](out[self.data.train_mask], true).item()
        }
        
        return metrics
    
    def evaluate(self, mask: str) -> Dict[str, float]:
        """Evaluate on given mask."""
        self.model.eval()
        
        with torch.no_grad():
            out = self.model(self.data)
            loss = F.nll_loss(out[getattr(self.data, f"{mask}_mask")], 
                            self.data.y[getattr(self.data, f"{mask}_mask")])
            
            pred = out[getattr(self.data, f"{mask}_mask")].argmax(dim=1)
            true = self.data.y[getattr(self.data, f"{mask}_mask")]
            
            metrics = {
                "loss": loss.item(),
                "accuracy": self.val_metrics["accuracy"](pred, true).item(),
                "f1": self.val_metrics["f1"](pred, true).item(),
                "auroc": self.val_metrics["auroc"](out[getattr(self.data, f"{mask}_mask")], true).item()
            }
        
        return metrics
    
    def train(self) -> Dict[str, List[float]]:
        """Train the model."""
        self.logger.info("Starting training...")
        
        for epoch in tqdm(range(1, self.config.get("epochs", 200) + 1), desc="Training"):
            # Train
            train_metrics = self.train_epoch()
            
            # Evaluate
            val_metrics = self.evaluate("val")
            test_metrics = self.evaluate("test")
            
            # Update history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["test_loss"].append(test_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["test_acc"].append(test_metrics["accuracy"])
            self.history["train_f1"].append(train_metrics["f1"])
            self.history["val_f1"].append(val_metrics["f1"])
            self.history["test_f1"].append(test_metrics["f1"])
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()
            
            # Early stopping
            if self.early_stopping:
                if self.early_stopping(val_metrics["loss"], self.model):
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Logging
            if epoch % self.config.get("log_interval", 20) == 0:
                self.logger.info(
                    f"Epoch {epoch:03d}: "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Test Loss: {test_metrics['loss']:.4f}, "
                    f"Train Acc: {train_metrics['accuracy']:.4f}, "
                    f"Val Acc: {val_metrics['accuracy']:.4f}, "
                    f"Test Acc: {test_metrics['accuracy']:.4f}"
                )
            
            # Save best model
            if self.config.get("save_best", True):
                if not self.best_metrics or val_metrics["accuracy"] > self.best_metrics.get("val_acc", 0):
                    self.best_metrics = {
                        "epoch": epoch,
                        "val_acc": val_metrics["accuracy"],
                        "val_f1": val_metrics["f1"],
                        "val_auroc": val_metrics["auroc"],
                        "test_acc": test_metrics["accuracy"],
                        "test_f1": test_metrics["f1"],
                        "test_auroc": test_metrics["auroc"]
                    }
                    
                    if self.config.get("checkpoint_dir"):
                        save_checkpoint(
                            self.model,
                            self.optimizer,
                            epoch,
                            val_metrics["loss"],
                            val_metrics,
                            f"{self.config['checkpoint_dir']}/best_model.pt"
                        )
        
        self.logger.info("Training completed!")
        if self.best_metrics:
            self.logger.info(f"Best validation accuracy: {self.best_metrics['val_acc']:.4f}")
            self.logger.info(f"Best test accuracy: {self.best_metrics['test_acc']:.4f}")
        
        return self.history
    
    def load_best_model(self, checkpoint_path: str):
        """Load the best model checkpoint."""
        checkpoint = load_checkpoint(self.model, self.optimizer, checkpoint_path)
        self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint
