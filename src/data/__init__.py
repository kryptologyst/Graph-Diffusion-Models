"""Data handling and preprocessing for graph diffusion models."""

import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import Planetoid, CoraFull, CiteSeer, PubMed, CoauthorCS, CoauthorPhysics
from torch_geometric.transforms import GDC, NormalizeFeatures, RandomNodeSplit
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import logging


class GraphDiffusionDataset(Dataset):
    """Custom dataset class for graph diffusion experiments."""
    
    def __init__(
        self,
        root: str,
        name: str = "Cora",
        transform: Optional[Any] = None,
        pre_transform: Optional[Any] = None,
        use_diffusion: bool = True,
        diffusion_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the dataset.
        
        Args:
            root: Root directory for the dataset
            name: Name of the dataset
            transform: Transform to apply to data
            pre_transform: Pre-transform to apply to data
            use_diffusion: Whether to apply diffusion preprocessing
            diffusion_config: Configuration for diffusion preprocessing
        """
        self.name = name
        self.use_diffusion = use_diffusion
        self.diffusion_config = diffusion_config or {}
        
        super().__init__(root, transform, pre_transform)
    
    @property
    def raw_file_names(self) -> List[str]:
        """Return list of raw file names."""
        return []
    
    @property
    def processed_file_names(self) -> List[str]:
        """Return list of processed file names."""
        return [f"{self.name.lower()}_diffusion.pt"]
    
    def download(self):
        """Download the dataset."""
        pass
    
    def process(self):
        """Process the dataset."""
        # Load the appropriate dataset
        dataset_classes = {
            "Cora": Planetoid,
            "CiteSeer": CiteSeer,
            "PubMed": PubMed,
            "CoraFull": CoraFull,
            "CoauthorCS": CoauthorCS,
            "CoauthorPhysics": CoauthorPhysics
        }
        
        if self.name not in dataset_classes:
            raise ValueError(f"Unknown dataset: {self.name}. Available: {list(dataset_classes.keys())}")
        
        # Load dataset
        dataset = dataset_classes[self.name](root=self.root, name=self.name)
        data = dataset[0]
        
        # Apply diffusion preprocessing if requested
        if self.use_diffusion:
            diffusion_transform = GDC(**self.diffusion_config)
            data = diffusion_transform(data)
        
        # Save processed data
        torch.save(data, self.processed_paths[0])
    
    def len(self) -> int:
        """Return the number of graphs in the dataset."""
        return 1
    
    def get(self, idx: int) -> Data:
        """Get a graph by index."""
        return torch.load(self.processed_paths[0])


def load_dataset(
    name: str = "Cora",
    root: str = "data",
    use_diffusion: bool = True,
    diffusion_config: Optional[Dict[str, Any]] = None,
    normalize_features: bool = True,
    random_split: bool = True,
    train_split: float = 0.6,
    val_split: float = 0.2,
    test_split: float = 0.2
) -> Tuple[Data, Dict[str, Any]]:
    """Load and preprocess a graph dataset.
    
    Args:
        name: Name of the dataset
        root: Root directory for data
        use_diffusion: Whether to apply diffusion preprocessing
        diffusion_config: Configuration for diffusion preprocessing
        normalize_features: Whether to normalize features
        random_split: Whether to use random split
        train_split: Training split ratio
        val_split: Validation split ratio
        test_split: Test split ratio
        
    Returns:
        Tuple of (data, dataset_info)
    """
    logger = logging.getLogger("graph_diffusion")
    
    # Default diffusion configuration
    if diffusion_config is None:
        diffusion_config = {
            "self_loop_weight": 1,
            "normalization_in": "sym",
            "normalization_out": "col",
            "diffusion_kwargs": {"method": "ppr", "alpha": 0.15},
            "sparsification_kwargs": {"method": "threshold", "eps": 0.01}
        }
    
    # Create transforms
    transforms = []
    
    if normalize_features:
        transforms.append(NormalizeFeatures())
    
    if use_diffusion:
        transforms.append(GDC(**diffusion_config))
    
    if random_split:
        transforms.append(RandomNodeSplit(
            num_train_per_class=train_split,
            num_val=val_split,
            num_test=test_split
        ))
    
    # Load dataset
    try:
        dataset_classes = {
            "Cora": Planetoid,
            "CiteSeer": CiteSeer,
            "PubMed": PubMed,
            "CoraFull": CoraFull,
            "CoauthorCS": CoauthorCS,
            "CoauthorPhysics": CoauthorPhysics
        }
        
        if name not in dataset_classes:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(dataset_classes.keys())}")
        
        dataset = dataset_classes[name](root=root, name=name, transform=transforms[0] if len(transforms) == 1 else transforms)
        data = dataset[0]
        
        logger.info(f"Loaded dataset: {name}")
        logger.info(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
        logger.info(f"Features: {data.num_node_features}, Classes: {data.num_classes}")
        
        # Dataset info
        dataset_info = {
            "name": name,
            "num_nodes": data.num_nodes,
            "num_edges": data.num_edges,
            "num_features": data.num_node_features,
            "num_classes": data.num_classes,
            "has_edge_attr": data.edge_attr is not None,
            "has_edge_weight": data.edge_weight is not None,
            "is_undirected": not data.is_directed()
        }
        
        return data, dataset_info
        
    except Exception as e:
        logger.error(f"Failed to load dataset {name}: {e}")
        raise


def generate_synthetic_graph(
    num_nodes: int = 1000,
    num_classes: int = 7,
    num_features: int = 1433,
    graph_type: str = "sbm",
    **kwargs
) -> Tuple[Data, Dict[str, Any]]:
    """Generate a synthetic graph for testing.
    
    Args:
        num_nodes: Number of nodes
        num_classes: Number of classes
        num_features: Number of features per node
        graph_type: Type of synthetic graph (sbm, ba, er)
        **kwargs: Additional parameters for graph generation
        
    Returns:
        Tuple of (data, dataset_info)
    """
    logger = logging.getLogger("graph_diffusion")
    
    if graph_type == "sbm":
        # Stochastic Block Model
        sizes = [num_nodes // num_classes] * num_classes
        sizes[-1] += num_nodes - sum(sizes)  # Adjust last block size
        
        probs = np.random.rand(num_classes, num_classes) * 0.3
        probs = (probs + probs.T) / 2  # Make symmetric
        np.fill_diagonal(probs, 0.8)  # Higher intra-block probability
        
        G = nx.stochastic_block_model(sizes, probs)
        
    elif graph_type == "ba":
        # Barabási-Albert
        m = kwargs.get("m", 5)
        G = nx.barabasi_albert_graph(num_nodes, m)
        
    elif graph_type == "er":
        # Erdős-Rényi
        p = kwargs.get("p", 0.01)
        G = nx.erdos_renyi_graph(num_nodes, p)
        
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")
    
    # Convert to PyTorch Geometric format
    data = from_networkx(G)
    
    # Add features
    data.x = torch.randn(num_nodes, num_features)
    
    # Add labels (random for now)
    data.y = torch.randint(0, num_classes, (num_nodes,))
    
    # Create train/val/test masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    # Random split
    indices = torch.randperm(num_nodes)
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    logger.info(f"Generated synthetic {graph_type} graph")
    logger.info(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    
    dataset_info = {
        "name": f"synthetic_{graph_type}",
        "num_nodes": data.num_nodes,
        "num_edges": data.num_edges,
        "num_features": data.num_node_features,
        "num_classes": data.num_classes,
        "has_edge_attr": data.edge_attr is not None,
        "has_edge_weight": data.edge_weight is not None,
        "is_undirected": not data.is_directed(),
        "graph_type": graph_type
    }
    
    return data, dataset_info


def analyze_graph(data: Data) -> Dict[str, Any]:
    """Analyze graph properties.
    
    Args:
        data: PyTorch Geometric data object
        
    Returns:
        Dictionary of graph statistics
    """
    stats = {
        "num_nodes": data.num_nodes,
        "num_edges": data.num_edges,
        "num_features": data.num_node_features,
        "num_classes": data.num_classes,
        "density": 2 * data.num_edges / (data.num_nodes * (data.num_nodes - 1)),
        "is_directed": data.is_directed(),
        "has_self_loops": data.has_self_loops(),
        "is_undirected": not data.is_directed()
    }
    
    # Degree statistics
    if hasattr(data, 'edge_index') and data.edge_index is not None:
        degrees = torch.zeros(data.num_nodes, dtype=torch.long)
        degrees.scatter_add_(0, data.edge_index[0], torch.ones(data.edge_index.size(1)))
        if not data.is_directed():
            degrees.scatter_add_(0, data.edge_index[1], torch.ones(data.edge_index.size(1)))
        
        stats.update({
            "avg_degree": degrees.float().mean().item(),
            "max_degree": degrees.max().item(),
            "min_degree": degrees.min().item(),
            "degree_std": degrees.float().std().item()
        })
    
    # Class distribution
    if hasattr(data, 'y') and data.y is not None:
        class_counts = torch.bincount(data.y)
        stats.update({
            "class_distribution": class_counts.tolist(),
            "class_balance": (class_counts.min() / class_counts.max()).item()
        })
    
    return stats
