"""Evaluation utilities for graph diffusion models."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from torchmetrics import Accuracy, F1Score, AUROC, Precision, Recall
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassF1Score, MulticlassAUROC,
    MulticlassPrecision, MulticlassRecall, MulticlassConfusionMatrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


class ModelEvaluator:
    """Comprehensive model evaluator for graph diffusion models."""
    
    def __init__(
        self,
        model: nn.Module,
        data,
        device: torch.device,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize evaluator.
        
        Args:
            model: Trained PyTorch model
            data: PyTorch Geometric data object
            device: Device to use for evaluation
            logger: Logger instance
        """
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = device
        self.logger = logger or logging.getLogger("graph_diffusion")
        
        # Setup metrics
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Setup evaluation metrics."""
        num_classes = self.data.num_classes
        
        self.metrics = {
            "accuracy": MulticlassAccuracy(num_classes=num_classes, average="micro"),
            "accuracy_macro": MulticlassAccuracy(num_classes=num_classes, average="macro"),
            "f1_micro": MulticlassF1Score(num_classes=num_classes, average="micro"),
            "f1_macro": MulticlassF1Score(num_classes=num_classes, average="macro"),
            "f1_weighted": MulticlassF1Score(num_classes=num_classes, average="weighted"),
            "auroc_macro": MulticlassAUROC(num_classes=num_classes, average="macro"),
            "precision_micro": MulticlassPrecision(num_classes=num_classes, average="micro"),
            "precision_macro": MulticlassPrecision(num_classes=num_classes, average="macro"),
            "recall_micro": MulticlassRecall(num_classes=num_classes, average="micro"),
            "recall_macro": MulticlassRecall(num_classes=num_classes, average="macro"),
            "confusion_matrix": MulticlassConfusionMatrix(num_classes=num_classes)
        }
    
    def evaluate_split(self, split: str) -> Dict[str, float]:
        """Evaluate model on a specific split.
        
        Args:
            split: Split name (train, val, test)
            
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get predictions
            out = self.model(self.data)
            mask = getattr(self.data, f"{split}_mask")
            
            if mask.sum() == 0:
                self.logger.warning(f"No samples in {split} split")
                return {}
            
            logits = out[mask]
            pred = logits.argmax(dim=1)
            true = self.data.y[mask]
            
            # Compute metrics
            metrics = {}
            for name, metric in self.metrics.items():
                if name == "confusion_matrix":
                    metrics[name] = metric(pred, true).cpu().numpy()
                else:
                    metrics[name] = metric(pred, true).item()
            
            # Add loss
            loss = nn.functional.nll_loss(logits, true)
            metrics["loss"] = loss.item()
            
            # Add per-class metrics
            metrics.update(self._compute_per_class_metrics(pred, true, logits))
            
        return metrics
    
    def _compute_per_class_metrics(self, pred: torch.Tensor, true: torch.Tensor, logits: torch.Tensor) -> Dict[str, List[float]]:
        """Compute per-class metrics.
        
        Args:
            pred: Predictions
            true: True labels
            logits: Logits
            
        Returns:
            Dictionary of per-class metrics
        """
        num_classes = self.data.num_classes
        
        # Per-class accuracy
        per_class_acc = []
        for i in range(num_classes):
            mask = true == i
            if mask.sum() > 0:
                acc = (pred[mask] == i).float().mean().item()
                per_class_acc.append(acc)
            else:
                per_class_acc.append(0.0)
        
        # Per-class F1
        per_class_f1 = []
        for i in range(num_classes):
            tp = ((pred == i) & (true == i)).sum().item()
            fp = ((pred == i) & (true != i)).sum().item()
            fn = ((pred != i) & (true == i)).sum().item()
            
            if tp + fp == 0:
                precision = 0.0
            else:
                precision = tp / (tp + fp)
            
            if tp + fn == 0:
                recall = 0.0
            else:
                recall = tp / (tp + fn)
            
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            
            per_class_f1.append(f1)
        
        return {
            "per_class_accuracy": per_class_acc,
            "per_class_f1": per_class_f1
        }
    
    def evaluate_all_splits(self) -> Dict[str, Dict[str, float]]:
        """Evaluate model on all splits.
        
        Returns:
            Dictionary of metrics for each split
        """
        results = {}
        
        for split in ["train", "val", "test"]:
            metrics = self.evaluate_split(split)
            if metrics:
                results[split] = metrics
                self.logger.info(f"{split.capitalize()} metrics: "
                               f"Acc: {metrics['accuracy']:.4f}, "
                               f"F1: {metrics['f1_macro']:.4f}, "
                               f"AUROC: {metrics['auroc_macro']:.4f}")
        
        return results
    
    def get_predictions(self, split: str = "test") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get predictions for a specific split.
        
        Args:
            split: Split name
            
        Returns:
            Tuple of (logits, predictions, true_labels)
        """
        self.model.eval()
        
        with torch.no_grad():
            out = self.model(self.data)
            mask = getattr(self.data, f"{split}_mask")
            
            logits = out[mask]
            pred = logits.argmax(dim=1)
            true = self.data.y[mask]
            
        return logits, pred, true
    
    def plot_confusion_matrix(self, split: str = "test", save_path: Optional[str] = None):
        """Plot confusion matrix.
        
        Args:
            split: Split name
            save_path: Path to save the plot
        """
        metrics = self.evaluate_split(split)
        if "confusion_matrix" not in metrics:
            return
        
        cm = metrics["confusion_matrix"]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=range(self.data.num_classes),
                   yticklabels=range(self.data.num_classes))
        plt.title(f'Confusion Matrix - {split.capitalize()} Split')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_per_class_metrics(self, split: str = "test", save_path: Optional[str] = None):
        """Plot per-class metrics.
        
        Args:
            split: Split name
            save_path: Path to save the plot
        """
        metrics = self.evaluate_split(split)
        if "per_class_accuracy" not in metrics:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Per-class accuracy
        classes = range(self.data.num_classes)
        acc = metrics["per_class_accuracy"]
        
        ax1.bar(classes, acc)
        ax1.set_title(f'Per-Class Accuracy - {split.capitalize()} Split')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        
        # Per-class F1
        f1 = metrics["per_class_f1"]
        
        ax2.bar(classes, f1)
        ax2.set_title(f'Per-Class F1 Score - {split.capitalize()} Split')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('F1 Score')
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_classification_report(self, split: str = "test") -> str:
        """Generate detailed classification report.
        
        Args:
            split: Split name
            
        Returns:
            Classification report string
        """
        _, pred, true = self.get_predictions(split)
        
        return classification_report(
            true.cpu().numpy(),
            pred.cpu().numpy(),
            target_names=[f'Class {i}' for i in range(self.data.num_classes)]
        )
    
    def analyze_errors(self, split: str = "test", top_k: int = 10) -> Dict[str, Any]:
        """Analyze prediction errors.
        
        Args:
            split: Split name
            top_k: Number of top errors to analyze
            
        Returns:
            Dictionary of error analysis
        """
        _, pred, true = self.get_predictions(split)
        
        # Find errors
        errors = pred != true
        error_indices = torch.where(errors)[0]
        
        if len(error_indices) == 0:
            return {"num_errors": 0, "error_rate": 0.0}
        
        # Analyze error patterns
        error_analysis = {
            "num_errors": len(error_indices),
            "error_rate": len(error_indices) / len(true),
            "most_confused_pairs": self._get_confused_pairs(pred, true),
            "error_by_class": self._get_error_by_class(pred, true)
        }
        
        return error_analysis
    
    def _get_confused_pairs(self, pred: torch.Tensor, true: torch.Tensor) -> List[Tuple[int, int, int]]:
        """Get most confused class pairs.
        
        Args:
            pred: Predictions
            true: True labels
            
        Returns:
            List of (true_class, pred_class, count) tuples
        """
        errors = pred != true
        error_pred = pred[errors]
        error_true = true[errors]
        
        # Count confusion pairs
        confusion_counts = {}
        for t, p in zip(error_true, error_pred):
            pair = (t.item(), p.item())
            confusion_counts[pair] = confusion_counts.get(pair, 0) + 1
        
        # Sort by count
        sorted_pairs = sorted(confusion_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [(t, p, count) for (t, p), count in sorted_pairs[:10]]
    
    def _get_error_by_class(self, pred: torch.Tensor, true: torch.Tensor) -> Dict[int, Dict[str, int]]:
        """Get error statistics by class.
        
        Args:
            pred: Predictions
            true: True labels
            
        Returns:
            Dictionary of error statistics per class
        """
        error_stats = {}
        
        for class_id in range(self.data.num_classes):
            class_mask = true == class_id
            class_pred = pred[class_mask]
            class_true = true[class_mask]
            
            total = len(class_true)
            correct = (class_pred == class_true).sum().item()
            errors = total - correct
            
            error_stats[class_id] = {
                "total": total,
                "correct": correct,
                "errors": errors,
                "accuracy": correct / total if total > 0 else 0.0
            }
        
        return error_stats


def create_model_leaderboard(results: Dict[str, Dict[str, Any]], save_path: Optional[str] = None) -> pd.DataFrame:
    """Create a model leaderboard.
    
    Args:
        results: Dictionary of model results
        save_path: Path to save the leaderboard
        
    Returns:
        DataFrame with model comparison
    """
    leaderboard_data = []
    
    for model_name, model_results in results.items():
        test_metrics = model_results.get("test", {})
        
        leaderboard_data.append({
            "Model": model_name,
            "Test Accuracy": test_metrics.get("accuracy", 0.0),
            "Test F1 (Macro)": test_metrics.get("f1_macro", 0.0),
            "Test F1 (Micro)": test_metrics.get("f1_micro", 0.0),
            "Test AUROC": test_metrics.get("auroc_macro", 0.0),
            "Test Loss": test_metrics.get("loss", float('inf')),
            "Parameters": model_results.get("num_parameters", 0)
        })
    
    df = pd.DataFrame(leaderboard_data)
    df = df.sort_values("Test Accuracy", ascending=False)
    
    if save_path:
        df.to_csv(save_path, index=False)
    
    return df
