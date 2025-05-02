"""
arxiv_dummy_train.py
Huub Al 

This script trains a SetGNN model on the dummy hypergraph to predict whether an unseen edge (author-paper connection) is plausible.
"""
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
import numpy as np
import argparse
import torch.nn.functional as F

# Add AllSet to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'AllSet', 'src'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))

from hypergraph_dummy import DummyHyperGraph
from models import SetGNN
from preprocessing import norm_contruction, ExtractV2E


class EdgePredictionModel(nn.Module):
    """
    Model for predicting edge plausibility
    """
    def __init__(self, setgnn_model):
        super().__init__()
        self.setgnn = setgnn_model
        self.edge_predictor = nn.Sequential(
            nn.Linear(64, 256),  # Node features from SetGNN
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
    
    def forward(self, data):
        # Get node embeddings from SetGNN before classification
        x, edge_index, norm = data.x, data.edge_index, data.norm
        x = F.dropout(x, p=0.2, training=self.training)  # Input dropout
        for i, _ in enumerate(self.setgnn.V2EConvs):
            x = F.relu(self.setgnn.V2EConvs[i](x, edge_index, norm, self.setgnn.aggr))
            x = F.dropout(x, p=self.setgnn.dropout, training=self.training)
            x = F.relu(self.setgnn.E2VConvs[i](x, torch.stack([edge_index[1], edge_index[0]], dim=0), norm, self.setgnn.aggr))
            x = F.dropout(x, p=self.setgnn.dropout, training=self.training)
        
        # Edge prediction
        edge_preds = []
        for i, (src, dst) in enumerate(data.edge_index.t()):
            if dst >= data.n_x[0]:  # If this is a hyperedge
                node_feat = x[src]  # Get node features
                edge_preds.append(self.edge_predictor(node_feat))
        
        return torch.cat(edge_preds)


def train(args):
    # Create dummy hypergraph
    print("Creating dummy hypergraph...")
    graph = DummyHyperGraph(
        num_authors=args.num_authors,
        num_papers=args.num_papers,
        feature_dim=args.num_features
    )
    data = graph.get_data()
    
    # Extract V2E and add normalization
    data = ExtractV2E(data)
    data = norm_contruction(data, option=args.normtype)
    
    # Create SetGNN model
    setgnn_model = SetGNN(args)
    setgnn_model = setgnn_model.to(args.device)
    
    # Create edge prediction model
    model = EdgePredictionModel(setgnn_model)
    model = model.to(args.device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    # Create loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    print("Starting training...")
    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        
        # Forward pass
        edge_preds = model(data)
        
        # Create edge labels (1 for existing edges, 0 for non-existent)
        edge_labels = torch.ones(edge_preds.size(0), device=args.device)
        
        # Compute loss
        loss = criterion(edge_preds, edge_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if epoch % args.display_step == 0:
            print(f'Epoch {epoch:03d}, Loss: {loss.item():.4f}')
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'setgnn_state_dict': setgnn_model.state_dict(),
        'args': args
    }, args.model_path)
    print(f"Model saved to {args.model_path}")


def main():
    parser = argparse.ArgumentParser()
    # Dataset arguments
    parser.add_argument('--num_authors', type=int, default=1000,
                      help='Number of authors in dummy graph')
    parser.add_argument('--num_papers', type=int, default=500,
                      help='Number of papers in dummy graph')
    parser.add_argument('--num_features', type=int, default=768,
                      help='Dimension of node features')
    parser.add_argument('--num_classes', type=int, default=1,
                      help='Number of classes (default: 1 for binary edge prediction)')
    
    # Training arguments
    parser.add_argument('--model_path', type=str, default='weights/dummy_model.pt',
                      help='Path to save the model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.0,
                      help='Weight decay')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of epochs')
    parser.add_argument('--display_step', type=int, default=10,
                      help='Display step')
    
    # Model architecture arguments
    parser.add_argument('--method', default='AllSetTransformer',
                      help='Model method')
    parser.add_argument('--All_num_layers', type=int, default=2,
                      help='Number of layers in AllSet')
    parser.add_argument('--MLP_num_layers', type=int, default=2,
                      help='Number of layers in MLP')
    parser.add_argument('--MLP_hidden', type=int, default=64,
                      help='Hidden dimension for MLP')
    parser.add_argument('--Classifier_num_layers', type=int, default=2,
                      help='Number of layers in classifier')
    parser.add_argument('--Classifier_hidden', type=int, default=64,
                      help='Hidden dimension for classifier')
    parser.add_argument('--heads', type=int, default=4,
                      help='Number of attention heads')
    parser.add_argument('--output_heads', type=int, default=1,
                      help='Number of output heads')
    parser.add_argument('--dropout', type=float, default=0.5,
                      help='Dropout rate')
    
    # SetGNN specific arguments
    parser.add_argument('--aggregate', default='mean', choices=['sum', 'mean'],
                      help='Aggregation method')
    parser.add_argument('--normalization', default='ln', choices=['bn', 'ln', 'None'],
                      help='Normalization method')
    parser.add_argument('--deepset_input_norm', action='store_true',
                      help='Whether to use input normalization')
    parser.add_argument('--GPR', action='store_true',
                      help='Whether to use GPR')
    parser.add_argument('--LearnMask', action='store_true',
                      help='Whether to learn mask')
    parser.add_argument('--PMA', action='store_true',
                      help='Whether to use PMA attention')
    parser.add_argument('--add_self_loop', action='store_true',
                      help='Whether to add self loops')
    parser.add_argument('--exclude_self', action='store_true',
                      help='Whether to exclude self connections')
    parser.add_argument('--normtype', default='all_one', choices=['all_one', 'deg_half_sym'],
                      help='Normalization type')
    
    # Set defaults
    parser.set_defaults(PMA=True)
    parser.set_defaults(add_self_loop=True)
    parser.set_defaults(exclude_self=False)
    parser.set_defaults(GPR=False)
    parser.set_defaults(LearnMask=False)
    parser.set_defaults(deepset_input_norm=True)
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main() 