"""Configuration management for graph diffusion models."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path
import yaml


@dataclass
class DataConfig:
    """Data configuration."""
    dataset_name: str = "Cora"
    data_dir: str = "data"
    train_split: float = 0.6
    val_split: float = 0.2
    test_split: float = 0.2
    random_split: bool = True
    normalize_features: bool = True


@dataclass
class DiffusionConfig:
    """Graph diffusion configuration."""
    method: str = "ppr"  # ppr, heat, or adjacency
    alpha: float = 0.15
    eps: float = 0.01
    self_loop_weight: float = 1.0
    normalization_in: str = "sym"  # sym, col, or row
    normalization_out: str = "col"  # sym, col, or row
    sparsification_method: str = "threshold"  # threshold or topk
    topk_k: int = 64


@dataclass
class ModelConfig:
    """Model configuration."""
    model_type: str = "gcn"  # gcn, gat, sage, gin
    hidden_channels: int = 64
    num_layers: int = 2
    dropout: float = 0.5
    activation: str = "relu"
    use_batch_norm: bool = False
    use_residual: bool = False
    cached: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""
    lr: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 200
    patience: int = 50
    batch_size: int = 1
    early_stopping: bool = True
    save_best: bool = True
    checkpoint_dir: str = "checkpoints"


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    seed: int = 42
    device: str = "auto"  # auto, cuda, mps, cpu
    log_level: str = "INFO"
    use_wandb: bool = False
    project_name: str = "graph_diffusion"
    experiment_name: str = "default"
    tags: List[str] = field(default_factory=list)


@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig = field(default_factory=DataConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Config instance
        """
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create nested config objects
        config = cls()
        for section, values in config_dict.items():
            if hasattr(config, section):
                section_config = getattr(config, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
        
        return config
    
    def to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file.
        
        Args:
            config_path: Path to save YAML configuration file
        """
        config_dict = {
            'data': self.data.__dict__,
            'diffusion': self.diffusion.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'experiment': self.experiment.__dict__
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def update(self, **kwargs) -> None:
        """Update configuration with keyword arguments.
        
        Args:
            **kwargs: Configuration updates
        """
        for key, value in kwargs.items():
            if '.' in key:
                section, param = key.split('.', 1)
                if hasattr(self, section):
                    section_obj = getattr(self, section)
                    if hasattr(section_obj, param):
                        setattr(section_obj, param, value)
            else:
                if hasattr(self, key):
                    setattr(self, key, value)
