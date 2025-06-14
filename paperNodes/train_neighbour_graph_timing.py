import torch
import copy
import matplotlib.pyplot as plt
import numpy as np
import time

import sys
sys.path.append("/Users/huubal/scriptie/data")
sys.path.append("/Users/huubal/scriptie/paperNodes")
from paperNodes_graph import arXivHyperGraph
from model import arXivHGNN

def train_and_measure_time(model, data, train_mask, val_mask, epochs, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    start_time = time.process_time()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.embeddings.to(device), data.incidence.to(device))
        loss = criterion(out[train_mask], data.labels[train_mask].to(device))
        loss.backward()
        optimizer.step()

    end_time = time.process_time()
    training_time = end_time - start_time
    
    return training_time

def get_masks(num_nodes, val_ratio=0.2):
    indices = torch.randperm(num_nodes)
    split = int((1 - val_ratio) * num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[indices[:split]] = True
    val_mask[indices[split:]] = True
    return train_mask, val_mask

def run_timing_experiment(device='cpu'):
    print("\n===== Running Neighborhood Graph Timing Experiment =====")
    print(f"Using device: {device}")

    # Step 1: Initialize hypergraph and subgraph
    hypergraph = arXivHyperGraph("data/arxiv-data/subset_cs_20000.json.gz")
    subgraph = hypergraph.construct_subgraph(dropout=0.1)
    
    # Get initial graph size
    num_nodes_full = subgraph.embeddings.shape[0]
    print(f"Full graph size: {num_nodes_full} nodes")

    # Initialize model parameters
    in_dim = subgraph.embeddings.shape[1]
    out_dim = len(hypergraph.full_label_map)

    # Step 2: Train on full graph
    print("\nTraining model on full graph...")
    full_train_mask, full_val_mask = get_masks(num_nodes_full)
    full_model = arXivHGNN(in_dim, hidden_channels=128, out_channels=out_dim).to(device)
    full_time = train_and_measure_time(full_model, subgraph, full_train_mask, full_val_mask, 
                                     epochs=100, device=device)
    print(f"Full graph training time: {full_time:.2f} seconds")

    # Step 3: Remove outliers and create neighborhood graph
    print("\nRemoving outliers...")
    subgraph.remove_outliers(outlier_fraction=0.01)
    num_nodes_no_outliers = subgraph.embeddings.shape[0]
    print(f"Nodes after outlier removal: {num_nodes_no_outliers}")
    print(f"Removed {num_nodes_full - num_nodes_no_outliers} outliers")

    print("\nConstructing neighborhood graph...")
    neighborhood_graph = subgraph.construct_outlier_neighbourhood()
    if neighborhood_graph is None:
        print("No outliers found to construct neighborhood graph!")
        return None
    
    num_nodes_neighborhood = neighborhood_graph.embeddings.shape[0]
    print(f"Neighborhood graph size: {num_nodes_neighborhood} nodes")
    print(f"Neighborhood graph is {num_nodes_neighborhood/num_nodes_full*100:.1f}% the size of the full graph")

    print("\nTraining model on neighborhood graph...")
    neighborhood_train_mask, neighborhood_val_mask = get_masks(num_nodes_neighborhood)
    neighborhood_model = arXivHGNN(in_dim, hidden_channels=128, out_channels=out_dim).to(device)
    neighborhood_time = train_and_measure_time(neighborhood_model, neighborhood_graph, 
                                            neighborhood_train_mask, neighborhood_val_mask,
                                            epochs=100, device=device)
    print(f"Neighborhood graph training time: {neighborhood_time:.2f} seconds")

    # Calculate speedup
    speedup = full_time / neighborhood_time
    print(f"\nSpeedup factor: {speedup:.2f}x faster training on neighborhood graph")

    return {
        'full_graph': {
            'time': full_time,
            'nodes': num_nodes_full,
            'train_size': full_train_mask.sum().item()
        },
        'neighborhood': {
            'time': neighborhood_time,
            'nodes': num_nodes_neighborhood,
            'train_size': neighborhood_train_mask.sum().item()
        }
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run timing experiment
    timing_results = run_timing_experiment(device=device)
    if timing_results is None:
        return

    # Create figure with three subplots: training time, graph sizes, and speedup
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    approaches = ['Full Graph', 'Neighborhood Graph']
    
    # Training Time plot
    times = [timing_results['full_graph']['time'], timing_results['neighborhood']['time']]
    bars1 = ax1.bar(approaches, times, alpha=0.8)
    ax1.set_ylabel('CPU Time (seconds)')
    ax1.set_title('Training Time Comparison')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels for training time
    for bar, time in zip(bars1, times):
        ax1.text(bar.get_x() + bar.get_width()/2., time,
                f'{time:.1f}s', ha='center', va='bottom')

    # Graph Size plot
    sizes = [timing_results['full_graph']['nodes'], timing_results['neighborhood']['nodes']]
    bars2 = ax2.bar(approaches, sizes, alpha=0.8)
    ax2.set_ylabel('Number of Nodes')
    ax2.set_title('Graph Size Comparison')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels for graph sizes
    for bar, size in zip(bars2, sizes):
        ax2.text(bar.get_x() + bar.get_width()/2., size,
                f'{size:,}', ha='center', va='bottom')

    # Speedup plot
    speedup = timing_results['full_graph']['time'] / timing_results['neighborhood']['time']
    ax3.bar(['Speedup'], [speedup], alpha=0.8)
    ax3.set_ylabel('Speedup Factor')
    ax3.set_title('Training Speedup')
    ax3.grid(True, alpha=0.3)
    ax3.text(0, speedup, f'{speedup:.1f}x', ha='center', va='bottom')

    plt.suptitle('Training Time and Graph Size Comparison: Full Graph vs Neighborhood Graph')
    plt.tight_layout()
    plt.savefig("neighborhood_training_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main() 