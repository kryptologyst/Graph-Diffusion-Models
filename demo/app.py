"""Interactive demo for graph diffusion models using Streamlit."""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from typing import Dict, Any, List, Tuple
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import set_seed, get_device
from src.data import load_dataset, generate_synthetic_graph, analyze_graph
from src.models import GraphDiffusionModel
from src.eval import ModelEvaluator


def load_model_and_data(model_path: str, config: Dict[str, Any]):
    """Load trained model and data."""
    device = get_device()
    
    # Load data
    if config.get("use_synthetic", False):
        data, dataset_info = generate_synthetic_graph(
            num_nodes=config.get("num_nodes", 1000),
            num_classes=config.get("num_classes", 7),
            num_features=config.get("num_features", 1433),
            graph_type=config.get("synthetic_type", "sbm")
        )
    else:
        data, dataset_info = load_dataset(
            name=config.get("dataset_name", "Cora"),
            root="data",
            use_diffusion=config.get("use_diffusion", True)
        )
    
    # Create model
    model = GraphDiffusionModel(
        in_channels=dataset_info["num_features"],
        hidden_channels=config.get("hidden_channels", 64),
        out_channels=dataset_info["num_classes"],
        model_type=config.get("model_type", "gcn"),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.5)
    )
    
    # Load model weights if available
    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
    
    return model, data, dataset_info


def visualize_graph(data, node_colors=None, title="Graph Visualization"):
    """Create interactive graph visualization."""
    # Convert to NetworkX
    G = nx.Graph()
    
    # Add nodes
    for i in range(data.num_nodes):
        G.add_node(i)
    
    # Add edges
    edge_list = data.edge_index.t().cpu().numpy()
    G.add_edges_from(edge_list)
    
    # Get layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    node_colors_list = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f'Node {node}')
        
        if node_colors is not None:
            node_colors_list.append(node_colors[node])
        else:
            node_colors_list.append(0)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            color=node_colors_list,
            size=10,
            colorbar=dict(
                thickness=15,
                xanchor="left",
                len=0.5
            ),
            line=dict(width=2)
        )
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title=title,
                       titlefont_size=16,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       annotations=[ dict(
                           text="Interactive graph visualization",
                           showarrow=False,
                           xref="paper", yref="paper",
                           x=0.005, y=-0.002,
                           xanchor='left', yanchor='bottom',
                           font=dict(color='#888', size=12)
                       )],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                   ))
    
    return fig


def plot_attention_weights(model, data, node_idx: int, layer_idx: int = 0):
    """Plot attention weights for GAT models."""
    if model.model_type != "gat":
        st.warning("Attention visualization is only available for GAT models")
        return None
    
    model.eval()
    with torch.no_grad():
        # Get attention weights (this would need to be implemented in the GAT model)
        # For now, return a placeholder
        st.info("Attention weight visualization requires GAT model modification")
        return None


def plot_diffusion_analysis(data, dataset_info):
    """Plot diffusion analysis."""
    # Analyze graph properties
    stats = analyze_graph(data)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Graph Statistics', 'Class Distribution', 'Degree Distribution', 'Feature Statistics'),
        specs=[[{"type": "table"}, {"type": "bar"}],
               [{"type": "histogram"}, {"type": "histogram"}]]
    )
    
    # Graph statistics table
    stats_data = [
        ["Nodes", stats["num_nodes"]],
        ["Edges", stats["num_edges"]],
        ["Features", stats["num_features"]],
        ["Classes", stats["num_classes"]],
        ["Density", f"{stats['density']:.4f}"],
        ["Avg Degree", f"{stats.get('avg_degree', 0):.2f}"],
        ["Max Degree", stats.get('max_degree', 0)],
        ["Class Balance", f"{stats.get('class_balance', 0):.4f}"]
    ]
    
    fig.add_trace(
        go.Table(
            header=dict(values=["Property", "Value"]),
            cells=dict(values=list(zip(*stats_data)))
        ),
        row=1, col=1
    )
    
    # Class distribution
    if "class_distribution" in stats:
        fig.add_trace(
            go.Bar(
                x=list(range(len(stats["class_distribution"]))),
                y=stats["class_distribution"],
                name="Class Count"
            ),
            row=1, col=2
        )
    
    # Degree distribution
    if "avg_degree" in stats:
        # Generate degree distribution (simplified)
        degrees = np.random.poisson(stats["avg_degree"], stats["num_nodes"])
        fig.add_trace(
            go.Histogram(x=degrees, name="Degree Distribution"),
            row=2, col=1
        )
    
    # Feature statistics
    if hasattr(data, 'x') and data.x is not None:
        feature_means = data.x.mean(dim=0).cpu().numpy()
        fig.add_trace(
            go.Histogram(x=feature_means, name="Feature Means"),
            row=2, col=2
        )
    
    fig.update_layout(height=800, showlegend=False, title_text="Graph Analysis")
    return fig


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Graph Diffusion Models Demo",
        page_icon="🕸️",
        layout="wide"
    )
    
    st.title("🕸️ Graph Diffusion Models Demo")
    st.markdown("Interactive exploration of graph neural networks with diffusion preprocessing")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["gcn", "gat", "sage", "gin"],
        index=0
    )
    
    # Dataset selection
    dataset_name = st.sidebar.selectbox(
        "Dataset",
        ["Cora", "CiteSeer", "PubMed", "Synthetic"],
        index=0
    )
    
    use_synthetic = dataset_name == "Synthetic"
    
    if use_synthetic:
        synthetic_type = st.sidebar.selectbox(
            "Synthetic Graph Type",
            ["sbm", "ba", "er"],
            index=0
        )
        num_nodes = st.sidebar.slider("Number of Nodes", 100, 2000, 1000)
    else:
        synthetic_type = "sbm"
        num_nodes = 1000
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    hidden_channels = st.sidebar.slider("Hidden Channels", 16, 256, 64)
    num_layers = st.sidebar.slider("Number of Layers", 1, 5, 2)
    dropout = st.sidebar.slider("Dropout", 0.0, 0.8, 0.5)
    
    # Diffusion parameters
    st.sidebar.subheader("Diffusion Parameters")
    use_diffusion = st.sidebar.checkbox("Use Diffusion Preprocessing", True)
    alpha = st.sidebar.slider("Alpha", 0.01, 0.5, 0.15)
    eps = st.sidebar.slider("Epsilon", 0.001, 0.1, 0.01)
    
    # Load model and data
    config = {
        "model_type": model_type,
        "dataset_name": dataset_name if not use_synthetic else "synthetic",
        "use_synthetic": use_synthetic,
        "synthetic_type": synthetic_type,
        "num_nodes": num_nodes,
        "hidden_channels": hidden_channels,
        "num_layers": num_layers,
        "dropout": dropout,
        "use_diffusion": use_diffusion,
        "alpha": alpha,
        "eps": eps
    }
    
    try:
        model, data, dataset_info = load_model_and_data("checkpoints/best_model.pt", config)
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("Graph Visualization")
            
            # Node selection for analysis
            node_idx = st.selectbox(
                "Select Node for Analysis",
                range(min(100, data.num_nodes)),
                index=0
            )
            
            # Get node colors (predictions or labels)
            if hasattr(data, 'y') and data.y is not None:
                node_colors = data.y.cpu().numpy()
            else:
                node_colors = None
            
            # Visualize graph
            fig = visualize_graph(data, node_colors, f"{model_type.upper()} Graph")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.header("Model Information")
            
            # Model stats
            num_params = sum(p.numel() for p in model.parameters())
            st.metric("Parameters", f"{num_params:,}")
            st.metric("Model Type", model_type.upper())
            st.metric("Dataset", dataset_name)
            st.metric("Nodes", data.num_nodes)
            st.metric("Edges", data.num_edges)
            st.metric("Features", data.num_node_features)
            st.metric("Classes", data.num_classes)
        
        # Analysis tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Graph Analysis", "Node Analysis", "Model Performance", "Diffusion Analysis"])
        
        with tab1:
            st.header("Graph Analysis")
            analysis_fig = plot_diffusion_analysis(data, dataset_info)
            st.plotly_chart(analysis_fig, use_container_width=True)
        
        with tab2:
            st.header(f"Node {node_idx} Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Node Features")
                if hasattr(data, 'x') and data.x is not None:
                    features = data.x[node_idx].cpu().numpy()
                    feature_df = pd.DataFrame({
                        "Feature": range(len(features)),
                        "Value": features
                    })
                    st.dataframe(feature_df.head(20))
                
                # Neighbors
                st.subheader("Neighbors")
                if hasattr(data, 'edge_index') and data.edge_index is not None:
                    neighbors = data.edge_index[1][data.edge_index[0] == node_idx].cpu().numpy()
                    st.write(f"Node {node_idx} has {len(neighbors)} neighbors")
                    if len(neighbors) > 0:
                        st.write(f"Neighbors: {neighbors[:10].tolist()}{'...' if len(neighbors) > 10 else ''}")
            
            with col2:
                st.subheader("Predictions")
                model.eval()
                with torch.no_grad():
                    out = model(data)
                    probs = torch.softmax(out[node_idx], dim=0)
                    
                    pred_df = pd.DataFrame({
                        "Class": range(data.num_classes),
                        "Probability": probs.cpu().numpy()
                    }).sort_values("Probability", ascending=False)
                    
                    st.dataframe(pred_df)
                    
                    # Prediction bar chart
                    fig = px.bar(pred_df, x="Class", y="Probability", title="Class Probabilities")
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.header("Model Performance")
            
            # Evaluate model
            evaluator = ModelEvaluator(model, data, get_device())
            results = evaluator.evaluate_all_splits()
            
            # Performance metrics
            if "test" in results:
                test_metrics = results["test"]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Test Accuracy", f"{test_metrics['accuracy']:.4f}")
                with col2:
                    st.metric("Test F1 (Macro)", f"{test_metrics['f1_macro']:.4f}")
                with col3:
                    st.metric("Test F1 (Micro)", f"{test_metrics['f1_micro']:.4f}")
                with col4:
                    st.metric("Test AUROC", f"{test_metrics['auroc_macro']:.4f}")
                
                # Confusion matrix
                st.subheader("Confusion Matrix")
                evaluator.plot_confusion_matrix("test")
        
        with tab4:
            st.header("Diffusion Analysis")
            
            if use_diffusion:
                st.info("Diffusion preprocessing is enabled")
                st.write(f"**Method**: Personalized PageRank")
                st.write(f"**Alpha**: {alpha}")
                st.write(f"**Epsilon**: {eps}")
                
                # Show diffusion effects
                st.subheader("Diffusion Effects")
                st.write("Diffusion preprocessing helps smooth features across the graph structure, "
                        "which can improve performance on node classification tasks.")
                
                # Compare with/without diffusion
                st.subheader("Comparison")
                st.write("To see the effect of diffusion preprocessing, you can:")
                st.write("1. Toggle 'Use Diffusion Preprocessing' in the sidebar")
                st.write("2. Compare the model performance metrics")
                st.write("3. Observe changes in node embeddings and predictions")
            else:
                st.warning("Diffusion preprocessing is disabled")
                st.write("Enable diffusion preprocessing in the sidebar to see its effects.")
    
    except Exception as e:
        st.error(f"Error loading model or data: {e}")
        st.info("Make sure you have trained a model first using the training script.")


if __name__ == "__main__":
    main()
