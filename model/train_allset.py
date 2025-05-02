import sys
import os

# add AllSet to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'AllSet', 'src'))
# Add project root and data folder to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_root, 'data')
if project_root not in sys.path:
    sys.path.append(project_root)
if data_path not in sys.path:
    sys.path.append(data_path)

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data

from hypergraph import ArxivHyperGraph
from models import SetGNN
import numpy as np
from sklearn.model_selection import train_test_split
import argparse

def create_data_object(graph):
    """
    Convert our hypergraph to PyTorch Geometric Data object for AllSet
    """
    # Get node features (paper embeddings)
    x = torch.tensor([paper_data['embedding'] for paper_data in graph.edge_dict.values()], dtype=torch.float32)
    
    # Create edge index (incidence matrix)
    edge_index = []
    num_nodes = len(graph.edge_dict)
    
    # Create hyperedge indices starting from num_nodes
    for paper_idx, (paper_id, paper_data) in enumerate(graph.edge_dict.items()):
        hyperedge_idx = num_nodes + paper_idx
        for author in paper_data['data']['authors'].split(', '):
            if author in graph.author_to_idx:
                edge_index.append([graph.author_to_idx[author], hyperedge_idx])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    
    # Create edge weights (all 1.0 for now)
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32)
    
    # Create Data object with required attributes
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        n_x=torch.tensor([num_nodes]),
        num_hyperedges=torch.tensor([len(graph.edge_dict)])
    )
    
    # Add normalization
    data.norm = edge_weight
    
    return data

def train(args):
    # Load hypergraph
    print("Loading hypergraph...")
    graph = ArxivHyperGraph(args.data_path)
    
    # Create data object
    data = create_data_object(graph)
    
    # Create model
    model = SetGNN(args)
    model = model.to(args.device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Training loop
    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data)
        
        # Compute loss (you'll need to define your loss function based on your task)
        loss = compute_loss(out, data)  # This needs to be implemented
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d}, Loss: {loss.item():.4f}')
    
    # Save model
    torch.save(model.state_dict(), args.model_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/arxiv-data/arxiv-metadata-small.json.gz',
                      help='Path to the arXiv data file')
    parser.add_argument('--model_path', type=str, default='models/allset_model.pt',
                      help='Path to save the model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                      help='Weight decay')
    parser.add_argument('--epochs', type=int, default=200,
                      help='Number of epochs')
    parser.add_argument('--hidden_dim', type=int, default=64,
                      help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                      help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                      help='Dropout rate')
    
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main() 