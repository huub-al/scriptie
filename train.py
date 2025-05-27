"""
train.py
Huub Al

Training script for InteractiveHGNN.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))

import torch
import torch.nn as nn
import torch.optim as optim
from hypergraph_tmx import ArxivHyperGraph
from model import InteractiveHGNN
import utils


def train_interactive_hgnn(input_file, num_epochs=15):
    """
    Train InteractiveHGNN on a mix of fake and real edges.
    
    Args:
        input_file (str): Path to the input JSON.GZ file
        num_epochs (int): Number of training epochs
    """
    # Initialize hypergraph and create subgraph
    print("Initializing hypergraph...")
    graph = ArxivHyperGraph(input_file)
    subgraph = graph.gen_subgraph(dropout=0.9)  # Keep 10% of edges
    node_features = subgraph.node_features
    n_edges = subgraph.sub_H.shape[1] + 1
    
    # Initialize model
    print("Initializing InteractiveHGNN...")
    model = InteractiveHGNN(n_edges)
    
    # Use CPU to keep sparse tensor efficiency
    device = torch.device('cpu')
    model = model.to(device)
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()  # Binary cross entropy for edge prediction
    
    # Generate training data
    print("Generating training data...")
    # Get real edges (dropped papers with authors)
    real_edges = subgraph.generate_real_edge()
    # Generate fake edges
    fake_edges = [subgraph.generate_fake_edge() for _ in range(100)]
    # Generate plausible fake edges
    plausible_fake_edges = [subgraph.generate_plausible_fake_edge() for _ in range(100)]
    
    # Create list of all edges with their target probabilities
    all_edges = []
    for edge in real_edges:
        all_edges.append((edge, 1.0))  # Real edges
    for edge in fake_edges:
        all_edges.append((edge, 0.0))  # Fake edges
    for edge in plausible_fake_edges:
        all_edges.append((edge, 0.5))  # Plausible fake edges
    
    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Shuffle edges for this epoch
        import random
        random.shuffle(all_edges)
        
        # Process each edge
        for edge, target_prob in all_edges:
            edge_embedding, authors = edge
            
            # Create target tensor
            target = torch.tensor([target_prob], device=device)
            
            # Add edge to incidence matrix
            # First, get current incidence matrix
            current_H = subgraph.sub_H.clone()
            
            # Create new column for the edge as a sparse tensor [N, 1]
            indices = torch.tensor([[idx, 0] for idx in authors], dtype=torch.long).t()
            values = torch.ones(len(authors), dtype=torch.float)
            new_col = torch.sparse_coo_tensor(indices, values, (current_H.shape[0], 1))

            # Append the new column using your custom utility
            new_H = utils.append_sparse_column(current_H, new_col, num_nodes=current_H.size(0))
            
            # Forward pass
            optimizer.zero_grad()
            edge_embedding = edge_embedding.to(device)
            prediction = model(node_features, new_H)
            
            # Calculate loss and backpropagate
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print epoch statistics
        avg_loss = total_loss / len(all_edges)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    print("Training completed!")
    return model

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python train.py <input_file>")
        sys.exit(1)
    
    train_interactive_hgnn(sys.argv[1]) 