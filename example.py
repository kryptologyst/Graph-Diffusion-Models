#!/usr/bin/env python3
"""Example script demonstrating basic usage of graph diffusion models."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
from src.utils import set_seed, get_device, setup_logging
from src.data import load_dataset, generate_synthetic_graph
from src.models import GraphDiffusionModel
from src.train import Trainer
from src.eval import ModelEvaluator


def main():
    """Run a simple example."""
    # Setup
    set_seed(42)
    device = get_device()
    logger = setup_logging("INFO")
    
    logger.info("Graph Diffusion Models - Example Usage")
    logger.info(f"Using device: {device}")
    
    # Load data (using synthetic data for this example)
    logger.info("Loading synthetic data...")
    data, dataset_info = generate_synthetic_graph(
        num_nodes=500,
        num_classes=5,
        num_features=100,
        graph_type="sbm"
    )
    
    logger.info(f"Dataset info: {dataset_info}")
    
    # Create model
    logger.info("Creating GCN model...")
    model = GraphDiffusionModel(
        in_channels=dataset_info["num_features"],
        hidden_channels=64,
        out_channels=dataset_info["num_classes"],
        model_type="gcn",
        num_layers=2,
        dropout=0.5
    )
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Training configuration
    training_config = {
        "lr": 0.01,
        "weight_decay": 5e-4,
        "epochs": 50,
        "patience": 20,
        "early_stopping": True,
        "log_interval": 10
    }
    
    # Train model
    logger.info("Starting training...")
    trainer = Trainer(
        model=model,
        data=data,
        config=training_config,
        device=device,
        logger=logger
    )
    
    history = trainer.train()
    
    # Evaluate model
    logger.info("Evaluating model...")
    evaluator = ModelEvaluator(model, data, device, logger)
    results = evaluator.evaluate_all_splits()
    
    # Print results
    logger.info("Results:")
    for split, metrics in results.items():
        logger.info(f"{split.capitalize()}: "
                   f"Acc: {metrics['accuracy']:.4f}, "
                   f"F1: {metrics['f1_macro']:.4f}, "
                   f"AUROC: {metrics['auroc_macro']:.4f}")
    
    # Generate classification report
    report = evaluator.generate_classification_report("test")
    logger.info("Classification Report:")
    logger.info(report)
    
    logger.info("Example completed successfully!")


if __name__ == "__main__":
    main()
