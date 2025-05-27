# train_allset.py

import torch


import sys
sys.path.append("/Users/huubal/scriptie/data")
from paperNodes_graph import arXivHyperGraph 
from model import arXivHGNN


def train(model, data, train_mask, val_mask, epochs=100, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.embeddings.to(device), data.incidence.to(device))
        loss = criterion(out[train_mask], data.labels[train_mask].to(device))
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        out = model(data.embeddings.to(device), data.incidence.to(device))
        pred = out.argmax(dim=1)
        correct = pred[val_mask].eq(data.labels[val_mask].to(device)).sum().item()
        acc = correct / val_mask.sum().item()
        return acc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Initialize hypergraph and subgraph
    print("Initializing Hypergraph...")
    hypergraph = arXivHyperGraph("data/arxiv-data/subset_cs_2000.json.gz")
    subgraph = hypergraph.construct_subgraph()

    # Step 2: Add 1000 papers
    print("Adding initial 1000 papers...")
    subgraph.add_papers(1000, fake=0.3, plausible=0.3)

    # Construct initial train/val mask
    num_nodes = subgraph.embeddings.shape[0]
    val_split = 0.2
    indices = torch.randperm(num_nodes)
    split = int((1 - val_split) * num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[indices[:split]] = True
    val_mask[indices[split:]] = True

    # Step 3: Initialize model
    in_dim = subgraph.embeddings.shape[1]
    out_dim = len(hypergraph.full_label_map)
    model = arXivHGNN(in_dim, hidden_channels=128, out_channels=out_dim).to(device)

    print("\nInitial Training...")
    acc = train(model, subgraph, train_mask, val_mask, epochs=100, device=device)
    print(f"Initial Validation Accuracy: {acc:.4f}")

    # Step 4: Iteratively add 100 papers, retrain, and report
    for i in range(10):
        print(f"\n--- Iteration {i+1} ---")
        prev_nodes = subgraph.embeddings.shape[0]

        print("Adding 100 new papers...")
        subgraph.add_papers(100, fake=0.3, plausible=0.3)

        new_nodes = subgraph.embeddings.shape[0]
        added_range = torch.arange(prev_nodes, new_nodes)

        # Extend masks: training only includes old and new nodes
        new_train_mask = torch.cat([
            train_mask,
            torch.ones(len(added_range), dtype=torch.bool)
        ])
        new_val_mask = torch.cat([
            val_mask,
            torch.zeros(len(added_range), dtype=torch.bool)
        ])

        print("Retraining model...")
        acc = train(model, subgraph, new_train_mask, new_val_mask, epochs=100, device=device)
        print(f"Validation Accuracy after adding 100 papers: {acc:.4f}")

        # Update masks for next iteration
        train_mask = new_train_mask
        val_mask = new_val_mask

if __name__ == "__main__":
    main()
