#!/usr/bin/env python3
"""Main training script for graph diffusion models."""

import argparse
import logging
import sys
from pathlib import Path
import torch
import wandb
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import set_seed, get_device, setup_logging
from src.utils.config import Config
from src.data import load_dataset, generate_synthetic_graph
from src.models import GraphDiffusionModel
from src.train import Trainer
from src.eval import ModelEvaluator, create_model_leaderboard


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train graph diffusion models")
    
    # Configuration
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--experiment-name", type=str, default=None,
                       help="Experiment name override")
    
    # Data
    parser.add_argument("--dataset", type=str, default=None,
                       help="Dataset name override")
    parser.add_argument("--synthetic", action="store_true",
                       help="Use synthetic data")
    parser.add_argument("--synthetic-type", type=str, default="sbm",
                       choices=["sbm", "ba", "er"],
                       help="Type of synthetic graph")
    parser.add_argument("--num-nodes", type=int, default=1000,
                       help="Number of nodes for synthetic graph")
    
    # Model
    parser.add_argument("--model", type=str, default=None,
                       choices=["gcn", "gat", "sage", "gin"],
                       help="Model type override")
    parser.add_argument("--hidden-channels", type=int, default=None,
                       help="Hidden channels override")
    parser.add_argument("--num-layers", type=int, default=None,
                       help="Number of layers override")
    
    # Training
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of epochs override")
    parser.add_argument("--lr", type=float, default=None,
                       help="Learning rate override")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size override")
    
    # Diffusion
    parser.add_argument("--no-diffusion", action="store_true",
                       help="Disable diffusion preprocessing")
    parser.add_argument("--alpha", type=float, default=None,
                       help="Diffusion alpha parameter")
    parser.add_argument("--eps", type=float, default=None,
                       help="Diffusion epsilon parameter")
    
    # Other
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed override")
    parser.add_argument("--device", type=str, default=None,
                       help="Device override")
    parser.add_argument("--wandb", action="store_true",
                       help="Enable wandb logging")
    parser.add_argument("--eval-only", action="store_true",
                       help="Only evaluate, don't train")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint for evaluation")
    
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Config:
    """Load configuration from file and command line arguments."""
    # Load base config
    config = Config.from_yaml(args.config)
    
    # Override with command line arguments
    overrides = {}
    
    if args.experiment_name:
        overrides["experiment.experiment_name"] = args.experiment_name
    
    if args.dataset:
        overrides["data.dataset_name"] = args.dataset
    
    if args.model:
        overrides["model.model_type"] = args.model
    
    if args.hidden_channels:
        overrides["model.hidden_channels"] = args.hidden_channels
    
    if args.num_layers:
        overrides["model.num_layers"] = args.num_layers
    
    if args.epochs:
        overrides["training.epochs"] = args.epochs
    
    if args.lr:
        overrides["training.lr"] = args.lr
    
    if args.batch_size:
        overrides["training.batch_size"] = args.batch_size
    
    if args.alpha:
        overrides["diffusion.alpha"] = args.alpha
    
    if args.eps:
        overrides["diffusion.eps"] = args.eps
    
    if args.seed:
        overrides["experiment.seed"] = args.seed
    
    if args.device:
        overrides["experiment.device"] = args.device
    
    if args.wandb:
        overrides["experiment.use_wandb"] = True
    
    # Apply overrides
    config.update(**overrides)
    
    return config


def setup_experiment(config: Config) -> Dict[str, Any]:
    """Setup experiment environment."""
    # Set random seed
    set_seed(config.experiment.seed)
    
    # Setup device
    if config.experiment.device == "auto":
        device = get_device()
    else:
        device = torch.device(config.experiment.device)
    
    # Setup logging
    logger = setup_logging(
        log_level=config.experiment.log_level,
        log_file=f"logs/{config.experiment.experiment_name}.log"
    )
    
    logger.info(f"Starting experiment: {config.experiment.experiment_name}")
    logger.info(f"Using device: {device}")
    logger.info(f"Configuration: {config}")
    
    # Setup wandb if enabled
    if config.experiment.use_wandb:
        wandb.init(
            project=config.experiment.project_name,
            name=config.experiment.experiment_name,
            tags=config.experiment.tags,
            config=config.__dict__
        )
    
    return {
        "device": device,
        "logger": logger
    }


def load_data(config: Config, args: argparse.Namespace) -> tuple:
    """Load or generate data."""
    logger = logging.getLogger("graph_diffusion")
    
    if args.synthetic:
        logger.info(f"Generating synthetic {args.synthetic_type} graph with {args.num_nodes} nodes")
        data, dataset_info = generate_synthetic_graph(
            num_nodes=args.num_nodes,
            num_classes=7,  # Default for Cora-like
            num_features=1433,  # Default for Cora-like
            graph_type=args.synthetic_type
        )
    else:
        logger.info(f"Loading dataset: {config.data.dataset_name}")
        
        # Diffusion configuration
        diffusion_config = None
        if not args.no_diffusion:
            diffusion_config = {
                "self_loop_weight": config.diffusion.self_loop_weight,
                "normalization_in": config.diffusion.normalization_in,
                "normalization_out": config.diffusion.normalization_out,
                "diffusion_kwargs": {
                    "method": config.diffusion.method,
                    "alpha": config.diffusion.alpha
                },
                "sparsification_kwargs": {
                    "method": config.diffusion.sparsification_method,
                    "eps": config.diffusion.eps
                }
            }
        
        data, dataset_info = load_dataset(
            name=config.data.dataset_name,
            root=config.data.data_dir,
            use_diffusion=not args.no_diffusion,
            diffusion_config=diffusion_config,
            normalize_features=config.data.normalize_features,
            random_split=config.data.random_split,
            train_split=config.data.train_split,
            val_split=config.data.val_split,
            test_split=config.data.test_split
        )
    
    logger.info(f"Dataset info: {dataset_info}")
    return data, dataset_info


def create_model(config: Config, dataset_info: Dict[str, Any]) -> GraphDiffusionModel:
    """Create model instance."""
    logger = logging.getLogger("graph_diffusion")
    
    model = GraphDiffusionModel(
        in_channels=dataset_info["num_features"],
        hidden_channels=config.model.hidden_channels,
        out_channels=dataset_info["num_classes"],
        model_type=config.model.model_type,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        activation=config.model.activation,
        use_batch_norm=config.model.use_batch_norm,
        use_residual=config.model.use_residual,
        cached=config.model.cached
    )
    
    logger.info(f"Created {config.model.model_type} model with {sum(p.numel() for p in model.parameters())} parameters")
    
    return model


def train_model(model: GraphDiffusionModel, data, config: Config, device: torch.device, logger: logging.Logger) -> Dict[str, Any]:
    """Train the model."""
    # Create trainer
    trainer = Trainer(
        model=model,
        data=data,
        config=config.training.__dict__,
        device=device,
        logger=logger
    )
    
    # Train
    history = trainer.train()
    
    # Get best metrics
    best_metrics = trainer.best_metrics
    
    return {
        "history": history,
        "best_metrics": best_metrics,
        "trainer": trainer
    }


def evaluate_model(model: GraphDiffusionModel, data, device: torch.device, logger: logging.Logger) -> Dict[str, Any]:
    """Evaluate the model."""
    evaluator = ModelEvaluator(model, data, device, logger)
    
    # Evaluate all splits
    results = evaluator.evaluate_all_splits()
    
    # Generate additional analysis
    analysis = {
        "classification_report": evaluator.generate_classification_report("test"),
        "error_analysis": evaluator.analyze_errors("test")
    }
    
    return {
        "results": results,
        "analysis": analysis,
        "evaluator": evaluator
    }


def main():
    """Main function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args)
    
    # Setup experiment
    setup_info = setup_experiment(config)
    device = setup_info["device"]
    logger = setup_info["logger"]
    
    try:
        # Load data
        data, dataset_info = load_data(config, args)
        
        # Create model
        model = create_model(config, dataset_info)
        
        if args.eval_only:
            # Load checkpoint if provided
            if args.checkpoint:
                checkpoint = torch.load(args.checkpoint, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                logger.info(f"Loaded checkpoint from {args.checkpoint}")
            
            # Evaluate only
            eval_results = evaluate_model(model, data, device, logger)
            logger.info("Evaluation completed")
            
        else:
            # Train model
            train_results = train_model(model, data, config, device, logger)
            
            # Evaluate model
            eval_results = evaluate_model(model, data, device, logger)
            
            # Log results
            logger.info("Training and evaluation completed")
            logger.info(f"Best validation accuracy: {train_results['best_metrics']['val_acc']:.4f}")
            logger.info(f"Test accuracy: {eval_results['results']['test']['accuracy']:.4f}")
            
            # Save results
            results_path = Path("results") / f"{config.experiment.experiment_name}_results.json"
            results_path.parent.mkdir(exist_ok=True)
            
            import json
            with open(results_path, "w") as f:
                json.dump({
                    "config": config.__dict__,
                    "dataset_info": dataset_info,
                    "train_results": train_results["best_metrics"],
                    "eval_results": eval_results["results"]
                }, f, indent=2, default=str)
            
            logger.info(f"Results saved to {results_path}")
    
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise
    
    finally:
        if config.experiment.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
