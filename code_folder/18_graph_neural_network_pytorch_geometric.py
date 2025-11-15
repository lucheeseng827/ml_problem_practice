"""
Graph Neural Networks with PyTorch Geometric
=============================================
Category 18: Multi-Modal - Learning on graph-structured data

Use cases: Social networks, molecules, knowledge graphs
"""

import numpy as np
import torch
import torch.nn as nn


class SimpleGCN(nn.Module):
    """Simple Graph Convolutional Network"""
    
    def __init__(self, in_features, hidden_features, out_features):
        super(SimpleGCN, self).__init__()
        self.conv1 = nn.Linear(in_features, hidden_features)
        self.conv2 = nn.Linear(hidden_features, out_features)
    
    def forward(self, x, adj):
        # First layer with adjacency aggregation
        x = torch.relu(adj @ self.conv1(x))
        # Second layer
        x = adj @ self.conv2(x)
        return x


def create_simple_graph():
    """Create a simple graph: 5 nodes, connected in a cycle"""
    # Node features (5 nodes, 3 features each)
    x = torch.randn(5, 3)
    
    # Adjacency matrix (cycle graph: 0-1-2-3-4-0)
    adj = torch.zeros(5, 5)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    for i, j in edges:
        adj[i, j] = 1
        adj[j, i] = 1
    
    # Add self-loops
    adj += torch.eye(5)
    
    # Normalize adjacency
    degrees = adj.sum(dim=1, keepdim=True)
    adj = adj / degrees
    
    return x, adj


def main():
    print("=" * 60)
    print("Graph Neural Networks (GNNs)")
    print("=" * 60)
    
    # Create graph
    node_features, adj_matrix = create_simple_graph()
    
    print(f"\nGraph: 5 nodes, 5 edges (cycle)")
    print(f"Node features shape: {node_features.shape}")
    print(f"Adjacency matrix shape: {adj_matrix.shape}")
    
    # Create GNN
    model = SimpleGCN(in_features=3, hidden_features=16, out_features=2)
    
    # Forward pass
    output = model(node_features, adj_matrix)
    
    print(f"\nGNN output shape: {output.shape}")
    print(f"Output (first 3 nodes):")
    print(output[:3])
    
    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("- GNNs aggregate information from neighbors")
    print("- Message passing neural networks")
    print("- Applications: social networks, molecules, citations")
    print("- PyTorch Geometric provides efficient GNN layers")
    print("- Can handle variable-size graphs")


if __name__ == "__main__":
    main()
