import torch
import copy
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("/Users/huubal/scriptie/data")
from paperNodes_graph import arXivHyperGraph, arXivSubGraph
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


def get_masks(num_nodes, val_ratio=0.2):
    indices = torch.randperm(num_nodes)
    split = int((1 - val_ratio) * num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[indices[:split]] = True
    val_mask[indices[split:]] = True
    return train_mask, val_mask


def run_experiment(repeat_id, learning_rates, device='cpu'):
    print(f"\n===== REPEAT {repeat_id + 1} =====")

    # Step 1: Initialize hypergraph and subgraph
    hypergraph = arXivHyperGraph("data/arxiv-data/subset_cs_20000.json.gz")
    base_subgraph = hypergraph.construct_subgraph(dropout=0.9)

    # Step 2: Train baseline
    num_nodes = base_subgraph.embeddings.shape[0]
    base_train_mask, base_val_mask = get_masks(num_nodes)

    in_dim = base_subgraph.embeddings.shape[1]
    out_dim = len(hypergraph.full_label_map)
    base_model = arXivHGNN(in_dim, hidden_channels=128, out_channels=out_dim).to(device)

    print("Training baseline model...")
    baseline_acc = train(base_model, base_subgraph, base_train_mask, base_val_mask, epochs=100, lr=1e-3, device=device)
    print(f"Baseline Accuracy: {baseline_acc:.4f}")

    # Save baseline model & subgraph state
    original_subgraph_state = copy.deepcopy(base_subgraph)
    original_model_state = copy.deepcopy(base_model.state_dict())
    original_train_mask = base_train_mask.clone()
    original_val_mask = base_val_mask.clone()

    # Step 3: Try expert learning rates
    accuracies_by_lr = {}

    for lr in learning_rates:
        print(f"\n--- Expert Simulation (LR={lr}) ---")
        accs = [baseline_acc]

        model = arXivHGNN(in_dim, hidden_channels=128, out_channels=out_dim).to(device)
        model.load_state_dict(copy.deepcopy(original_model_state))

        subgraph = copy.deepcopy(original_subgraph_state)
        train_mask = original_train_mask.clone()
        val_mask = original_val_mask.clone()

        for i in range(10):
            print(f"  Expert Iteration {i+1}: Adding 100 real papers...")
            prev_nodes = subgraph.embeddings.shape[0]
            subgraph.add_papers(100, fake=0.0, plausible=0.0)
            new_nodes = subgraph.embeddings.shape[0]

            # Extend training and keep original val mask
            train_mask = torch.cat([train_mask, torch.ones(new_nodes - prev_nodes, dtype=torch.bool)])
            val_mask = torch.cat([val_mask, torch.zeros(new_nodes - prev_nodes, dtype=torch.bool)])

            acc = train(model, subgraph, train_mask, val_mask, epochs=100, lr=lr, device=device)
            accs.append(acc)
            print(f"    Validation Accuracy: {acc:.4f}")

        accuracies_by_lr[lr] = accs

    return accuracies_by_lr


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rates = [5e-4, 1e-3, 5e-3]
    num_repeats = 5 

    # Initialize results container
    all_runs = {lr: [] for lr in learning_rates}

    for repeat in range(num_repeats):
        result = run_experiment(repeat, learning_rates, device=device)
        for lr in learning_rates:
            all_runs[lr].append(result[lr])  # List of accuracy lists per repeat

    # Compute averages
    avg_accs = {}
    std_accs = {}
    for lr in learning_rates:
        acc_matrix = np.array(all_runs[lr])  # Shape: (num_repeats, 11)
        avg_accs[lr] = acc_matrix.mean(axis=0)
        std_accs[lr] = acc_matrix.std(axis=0)

    # Plot averaged results
    plt.figure(figsize=(10, 6))
    for lr in learning_rates:
        avg = avg_accs[lr]
        std = std_accs[lr]
        x = np.arange(len(avg))
        plt.plot(x, avg, label=f"LR={lr}")
        plt.fill_between(x, avg - std, avg + std, alpha=0.2)
    plt.xlabel("Expert Iteration (0 = baseline)")
    plt.ylabel("Validation Accuracy on Original Set")
    plt.title(f"Expert Simulation: Mean ± Std Accuracy over {num_repeats} Runs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("expert_learning_curve_avg.png")
    plt.show()


if __name__ == "__main__":
    main()
