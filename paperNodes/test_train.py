import torch
import copy
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("/Users/huubal/scriptie/data")
sys.path.append("/Users/huubal/scriptie/paperNodes")
from paperNodes_graph import arXivHyperGraph
from model import arXivHGNN

def train(model, data, train_mask, val_mask, epochs, lr, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.embeddings.to(device), data.incidence.to(device))
        loss = criterion(out[train_mask], data.labels[train_mask].to(device))
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(data.embeddings.to(device), data.incidence.to(device))
        pred = out.argmax(dim=1)
        correct = pred[val_mask].eq(data.labels[val_mask].to(device)).sum().item()
        acc = correct / val_mask.sum().item()
    return acc

def get_masks(num_nodes, val_ratio=0.2, fixed_val_mask=None):
    if fixed_val_mask is not None:
        # Create train mask that excludes validation nodes
        train_mask = ~fixed_val_mask
        return train_mask, fixed_val_mask
    
    indices = torch.randperm(num_nodes)
    split = int((1 - val_ratio) * num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[indices[:split]] = True
    val_mask[indices[split:]] = True
    return train_mask, val_mask

def run_experiment(device='cpu'):
    print("\n===== Running Experiment =====")
    print(f"Using device: {device}")

    # Step 1: Initialize hypergraph and subgraph
    hypergraph = arXivHyperGraph("data/arxiv-data/subset_cs_20000.json.gz")
    original_subgraph = hypergraph.construct_subgraph(dropout=0.1)
    
    # Create train/val split for the full graph
    num_nodes = original_subgraph.embeddings.shape[0]
    train_mask, val_mask = get_masks(num_nodes)
    
    # Initialize model parameters
    in_dim = original_subgraph.embeddings.shape[1]
    out_dim = len(hypergraph.full_label_map)

    # Step 2: Train master model on full subgraph
    print("\nTraining master model on full subgraph...")
    master_model = arXivHGNN(in_dim, hidden_channels=128, out_channels=out_dim).to(device)
    master_acc = train(master_model, original_subgraph, train_mask, val_mask,
                      epochs=100, lr=1e-3, device=device)
    print(f"Master model accuracy: {master_acc:.4f}")

    # Step 3: Create a copy of the subgraph, remove outliers and train baseline model
    print("\nRemoving outliers and training baseline model...")
    subgraph_without_outliers = copy.deepcopy(original_subgraph)
    subgraph_without_outliers.remove_outliers(outlier_fraction=0.01)
    num_nodes_no_outliers = subgraph_without_outliers.embeddings.shape[0]
    print(f"Nodes after outlier removal: {num_nodes_no_outliers}")
    
    # Create new masks for clean data, preserving validation set from master model
    # We need to map the validation mask to the new graph structure
    clean_val_mask = val_mask[:num_nodes_no_outliers]
    clean_train_mask = ~clean_val_mask  # All non-validation nodes become training nodes
    
    # Train baseline model
    baseline_model = arXivHGNN(in_dim, hidden_channels=128, out_channels=out_dim).to(device)
    baseline_acc = train(baseline_model, subgraph_without_outliers, clean_train_mask, clean_val_mask,
                        epochs=100, lr=1e-3, device=device)
    print(f"Baseline model accuracy on clean data: {baseline_acc:.4f}")

    # Step 4: Create neighborhood graph for interactive training
    print("\nConstructing neighborhood graph for interactive training...")
    neighborhood_graph = subgraph_without_outliers.construct_outlier_neighbourhood()
    if neighborhood_graph is None:
        print("No outliers found to construct neighborhood graph!")
        return None
    
    num_nodes_neighborhood = neighborhood_graph.embeddings.shape[0]
    print(f"Neighborhood graph size: {num_nodes_neighborhood} nodes")
    
    # Create masks for neighborhood graph, preserving validation set from master model
    # We need to map the validation mask to the neighborhood graph structure
    neighborhood_val_mask = val_mask[:num_nodes_neighborhood]
    neighborhood_train_mask = ~neighborhood_val_mask  # All non-validation nodes become training nodes
    
    # Train interactive model
    print("\nTraining interactive model on neighborhood graph...")
    interactive_model = arXivHGNN(in_dim, hidden_channels=128, out_channels=out_dim).to(device)
    interactive_model.load_state_dict(copy.deepcopy(baseline_model.state_dict()))
    interactive_acc = train(interactive_model, neighborhood_graph, 
                          neighborhood_train_mask, neighborhood_val_mask,
                          epochs=15, lr=1e-4, device=device)
    print(f"Interactive model accuracy: {interactive_acc:.4f}")

    # Step 5: Evaluate baseline model on original validation set
    print("\nEvaluating baseline model on original validation set...")
    baseline_model.eval()
    with torch.no_grad():
        out = baseline_model(original_subgraph.embeddings.to(device), 
                           original_subgraph.incidence.to(device))
        pred = out.argmax(dim=1)
        correct = pred[val_mask].eq(original_subgraph.labels[val_mask].to(device)).sum().item()
        acc = correct / val_mask.sum().item()
    
    print(f"\nFinal Results:")
    print(f"Baseline model accuracy on original validation set: {acc:.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_experiment(device=device)

if __name__ == "__main__":
    main() 