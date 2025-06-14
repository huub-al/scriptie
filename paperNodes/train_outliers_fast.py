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

def evaluate_model(model, data_state, val_mask, device='cpu'):
    model.eval()
    with torch.no_grad():
        out = model(data_state['embeddings'].to(device), 
                   data_state['incidence'].to(device))
        pred = out.argmax(dim=1)
        correct = pred[val_mask].eq(data_state['labels'][val_mask].to(device)).sum().item()
        acc = correct / val_mask.sum().item()
    return acc

def run_comparison_experiment(device='cpu'):
    print("\n===== Running Comparison Experiment =====")
    print(f"Using device: {device}")

    # Step 1: Initialize hypergraph and subgraph
    hypergraph = arXivHyperGraph("data/arxiv-data/subset_cs_20000.json.gz")
    subgraph = hypergraph.construct_subgraph(dropout=0.1)
    
    # Create initial train/val split for the full graph
    num_nodes_full = subgraph.embeddings.shape[0]
    full_train_mask, full_val_mask = get_masks(num_nodes_full)
    
    # Save the full graph state and masks
    full_graph_state = {
        'embeddings': subgraph.embeddings.clone(),
        'labels': subgraph.labels.clone(),
        'incidence': subgraph.incidence.clone(),
        'node_to_authors': copy.deepcopy(subgraph.node_to_authors),
        'train_mask': full_train_mask.clone(),
        'val_mask': full_val_mask.clone()
    }
    
    # Initialize model parameters
    in_dim = subgraph.embeddings.shape[1]
    out_dim = len(hypergraph.full_label_map)

    # Step 2: Train master model on full graph
    print("\nTraining master model on full graph...")
    master_model = arXivHGNN(in_dim, hidden_channels=128, out_channels=out_dim).to(device)
    master_acc = train(master_model, subgraph, full_train_mask, full_val_mask, 
                      epochs=100, lr=1e-3, device=device)
    print(f"Master model accuracy: {master_acc:.4f}")

    # Step 3: Remove outliers and train baseline model
    print("\nRemoving outliers and training baseline model...")
    subgraph.remove_outliers(outlier_fraction=0.05)
    num_nodes_no_outliers = subgraph.embeddings.shape[0]
    print(f"Nodes after outlier removal: {num_nodes_no_outliers}")
    
    # Create masks for clean data, preserving validation set from master model
    clean_val_mask = full_val_mask[:num_nodes_no_outliers]
    clean_train_mask = ~clean_val_mask  # All non-validation nodes become training nodes
    
    # Train baseline model
    baseline_model = arXivHGNN(in_dim, hidden_channels=128, out_channels=out_dim).to(device)
    baseline_acc = train(baseline_model, subgraph, clean_train_mask, clean_val_mask,
                        epochs=100, lr=1e-3, device=device)
    print(f"Baseline model accuracy: {baseline_acc:.4f}")

    # Step 4: Create neighborhood graph for interactive training
    print("\nConstructing neighborhood graph for interactive training...")
    neighborhood_graph = subgraph.construct_outlier_neighbourhood()
    if neighborhood_graph is None:
        print("No outliers found to construct neighborhood graph!")
        return None
    
    num_nodes_neighborhood = neighborhood_graph.embeddings.shape[0]
    print(f"Neighborhood graph size: {num_nodes_neighborhood} nodes")
    
    # Save the neighborhood graph state
    neighborhood_graph_state = {
        'embeddings': neighborhood_graph.embeddings.clone(),
        'labels': neighborhood_graph.labels.clone(),
        'incidence': neighborhood_graph.incidence.clone(),
        'node_to_authors': copy.deepcopy(neighborhood_graph.node_to_authors)
    }
    
    # Create masks for neighborhood graph, preserving validation set from master model
    neighborhood_val_mask = full_val_mask[:num_nodes_neighborhood]
    neighborhood_train_mask = ~neighborhood_val_mask  # All non-validation nodes become training nodes
    
    # Test different learning rates and epochs for interactive model
    learning_rates = [1e-4, 5e-4, 1e-3]
    epochs_range = range(3, 25)
    interactive_results = {lr: [] for lr in learning_rates}
    
    for lr in learning_rates:
        print(f"\nTesting learning rate: {lr}")
        for epochs in epochs_range:
            print(f"Training with {epochs} epochs...")
            interactive_model = arXivHGNN(in_dim, hidden_channels=128, out_channels=out_dim).to(device)
            interactive_model.load_state_dict(copy.deepcopy(baseline_model.state_dict()))
            interactive_acc = train(interactive_model, neighborhood_graph, 
                                  neighborhood_train_mask, neighborhood_val_mask,
                                  epochs=epochs, lr=lr, device=device)
            interactive_results[lr].append(interactive_acc)
            print(f"Interactive model accuracy: {interactive_acc:.4f}")

    # Step 5: Evaluate all models on both validation sets
    print("\nEvaluating models on validation sets...")
    
    # Evaluate all models on full graph validation set
    master_full_acc = evaluate_model(master_model, full_graph_state, full_graph_state['val_mask'], device)
    baseline_full_acc = evaluate_model(baseline_model, full_graph_state, full_graph_state['val_mask'], device)
    
    # Evaluate all models on neighborhood graph validation set
    master_neighborhood_acc = evaluate_model(master_model, neighborhood_graph_state, neighborhood_val_mask, device)
    baseline_neighborhood_acc = evaluate_model(baseline_model, neighborhood_graph_state, neighborhood_val_mask, device)
    
    # Store results
    master_results = {
        'full_val_acc': master_full_acc,
        'neighborhood_acc': master_neighborhood_acc
    }
    baseline_results = {
        'full_val_acc': baseline_full_acc,
        'neighborhood_acc': baseline_neighborhood_acc
    }
    
    return {
        'interactive_results': interactive_results,
        'baseline_results': baseline_results,
        'master_results': master_results,
        'epochs_range': list(epochs_range)
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_repeats = 5

    # Initialize results containers
    all_interactive_results = {lr: [] for lr in [1e-4, 5e-4, 1e-3]}
    all_baseline_results = {'full_val_acc': [], 'neighborhood_acc': []}
    all_master_results = {'full_val_acc': [], 'neighborhood_acc': []}

    # Run experiments
    for repeat in range(num_repeats):
        print(f"\n===== REPEAT {repeat + 1}/{num_repeats} =====")
        results = run_comparison_experiment(device=device)
        if results is None:
            continue

        # Store results
        for lr in results['interactive_results']:
            all_interactive_results[lr].append(results['interactive_results'][lr])
        all_baseline_results['full_val_acc'].append(results['baseline_results']['full_val_acc'])
        all_baseline_results['neighborhood_acc'].append(results['baseline_results']['neighborhood_acc'])
        all_master_results['full_val_acc'].append(results['master_results']['full_val_acc'])
        all_master_results['neighborhood_acc'].append(results['master_results']['neighborhood_acc'])

    # Compute averages
    avg_interactive_results = {lr: np.mean(all_interactive_results[lr], axis=0) for lr in all_interactive_results}
    avg_baseline_results = {
        'full_val_acc': np.mean(all_baseline_results['full_val_acc']),
        'neighborhood_acc': np.mean(all_baseline_results['neighborhood_acc'])
    }
    avg_master_results = {
        'full_val_acc': np.mean(all_master_results['full_val_acc']),
        'neighborhood_acc': np.mean(all_master_results['neighborhood_acc'])
    }

    # Print average results
    print("\n" + "="*50)
    print("AVERAGE RESULTS")
    print("="*50)
    print("\nBaseline Model:")
    print(f"Full Graph Validation Accuracy: {avg_baseline_results['full_val_acc']:.4f}")
    print(f"Neighborhood Graph Validation Accuracy: {avg_baseline_results['neighborhood_acc']:.4f}")
    print("\nMaster Model:")
    print(f"Full Graph Validation Accuracy: {avg_master_results['full_val_acc']:.4f}")
    print(f"Neighborhood Graph Validation Accuracy: {avg_master_results['neighborhood_acc']:.4f}")

    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Full Graph Validation Accuracy
    epochs = results['epochs_range']
    learning_rates = list(avg_interactive_results.keys())
    
    # Plot interactive model results for each learning rate
    for lr in learning_rates:
        accuracies = avg_interactive_results[lr]
        ax1.plot(epochs, accuracies, marker='o', label=f'Interactive (lr={lr})')
    
    # Plot baseline and master model results as horizontal lines
    ax1.axhline(y=avg_baseline_results['full_val_acc'], 
                color='r', linestyle='--', 
                label=f'Baseline (100 epochs)')
    ax1.axhline(y=avg_master_results['full_val_acc'], 
                color='g', linestyle='--', 
                label=f'Master (100 epochs)')
    
    ax1.set_xlabel('Number of Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Full Graph Validation Accuracy')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Neighborhood Graph Validation Accuracy
    for lr in learning_rates:
        accuracies = avg_interactive_results[lr]
        ax2.plot(epochs, accuracies, marker='o', label=f'Interactive (lr={lr})')
    
    # Plot baseline and master model results as horizontal lines
    ax2.axhline(y=avg_baseline_results['neighborhood_acc'], 
                color='r', linestyle='--', 
                label=f'Baseline (100 epochs)')
    ax2.axhline(y=avg_master_results['neighborhood_acc'], 
                color='g', linestyle='--', 
                label=f'Master (100 epochs)')
    
    ax2.set_xlabel('Number of Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Neighborhood Graph Validation Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle('Model Performance Comparison: Full Graph vs Neighborhood Graph (Averaged over 5 runs)')
    plt.tight_layout()
    plt.savefig("interactive_training_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main() 