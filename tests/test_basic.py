"""Unit tests for graph diffusion models."""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils import set_seed, get_device, count_parameters
from src.utils.config import Config
from src.data import generate_synthetic_graph, analyze_graph
from src.models import GraphDiffusionModel


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        assert torch.initial_seed() == 42
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = torch.nn.Linear(10, 5)
        num_params = count_parameters(model)
        assert num_params == 55  # 10*5 + 5


class TestConfig:
    """Test configuration management."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = Config()
        assert config.data.dataset_name == "Cora"
        assert config.model.model_type == "gcn"
        assert config.training.lr == 0.01
    
    def test_config_update(self):
        """Test configuration updates."""
        config = Config()
        config.update(data={"dataset_name": "CiteSeer"})
        assert config.data.dataset_name == "CiteSeer"


class TestData:
    """Test data utilities."""
    
    def test_synthetic_graph_generation(self):
        """Test synthetic graph generation."""
        data, dataset_info = generate_synthetic_graph(
            num_nodes=100,
            num_classes=3,
            num_features=10,
            graph_type="sbm"
        )
        
        assert data.num_nodes == 100
        assert data.num_classes == 3
        assert data.num_node_features == 10
        assert dataset_info["graph_type"] == "sbm"
    
    def test_graph_analysis(self):
        """Test graph analysis."""
        data, _ = generate_synthetic_graph(num_nodes=50, num_classes=2)
        stats = analyze_graph(data)
        
        assert "num_nodes" in stats
        assert "num_edges" in stats
        assert "density" in stats


class TestModels:
    """Test model implementations."""
    
    def test_gcn_model(self):
        """Test GCN model."""
        model = GraphDiffusionModel(
            in_channels=10,
            hidden_channels=32,
            out_channels=3,
            model_type="gcn"
        )
        
        # Create dummy data
        x = torch.randn(20, 10)
        edge_index = torch.randint(0, 20, (2, 30))
        edge_weight = torch.randn(30)
        
        # Test forward pass
        out = model.gnn(x, edge_index, edge_weight)
        assert out.shape == (20, 3)
    
    def test_gat_model(self):
        """Test GAT model."""
        model = GraphDiffusionModel(
            in_channels=10,
            hidden_channels=32,
            out_channels=3,
            model_type="gat",
            heads=4
        )
        
        # Create dummy data
        x = torch.randn(20, 10)
        edge_index = torch.randint(0, 20, (2, 30))
        
        # Test forward pass
        out = model.gnn(x, edge_index)
        assert out.shape == (20, 3)
    
    def test_sage_model(self):
        """Test GraphSAGE model."""
        model = GraphDiffusionModel(
            in_channels=10,
            hidden_channels=32,
            out_channels=3,
            model_type="sage"
        )
        
        # Create dummy data
        x = torch.randn(20, 10)
        edge_index = torch.randint(0, 20, (2, 30))
        
        # Test forward pass
        out = model.gnn(x, edge_index)
        assert out.shape == (20, 3)
    
    def test_gin_model(self):
        """Test GIN model."""
        model = GraphDiffusionModel(
            in_channels=10,
            hidden_channels=32,
            out_channels=3,
            model_type="gin"
        )
        
        # Create dummy data
        x = torch.randn(20, 10)
        edge_index = torch.randint(0, 20, (2, 30))
        
        # Test forward pass
        out = model.gnn(x, edge_index)
        assert out.shape == (20, 3)


if __name__ == "__main__":
    pytest.main([__file__])
