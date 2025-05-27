# train_allset.py

import torch

import sys
sys.path.append("/Users/huubal/scriptie/data")
from paperNodes_graph import arXivHyperGraph 
from model import arXivHGNN
from collections import defaultdict


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

        if epoch % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_out = model(data.embeddings.to(device), data.incidence.to(device))
                pred = val_out.argmax(dim=1)
                correct = pred[val_mask].eq(data.labels[val_mask].to(device)).sum().item()
                acc = correct / val_mask.sum().item()
                print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val Acc: {acc:.4f}")
            model.train()

    return model


def evaluate_per_label(model, data, val_mask, label_map, device='cpu'):
    model.eval()
    with torch.no_grad():
        out = model(data.embeddings.to(device), data.incidence.to(device))
        preds = out.argmax(dim=1).cpu()
        labels = data.labels.cpu()

        counts = defaultdict(int)
        corrects = defaultdict(int)

        for i in torch.where(val_mask)[0]:
            true_label = labels[i].item()
            pred_label = preds[i].item()
            counts[true_label] += 1
            if pred_label == true_label:
                corrects[true_label] += 1

        print("\n📊 Per-label Accuracy:")
        idx_to_label = {v: k for k, v in label_map.items()}
        for label_idx in sorted(counts.keys()):
            label_name = idx_to_label.get(label_idx, f"Label {label_idx}")
            total = counts[label_idx]
            correct = corrects[label_idx]
            acc = correct / total if total > 0 else 0.0
            print(f"  - {label_name:<30}: {acc:.4f} ({correct}/{total})")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading arXiv HyperGraph...")
    hypergraph = arXivHyperGraph()
    subgraph = hypergraph.construct_subgraph()

    print("Adding 1000 papers...")
    subgraph.add_papers(10009, fake=0.3, plausible=0.3)

    num_nodes = subgraph.embeddings.shape[0]
    indices = torch.randperm(num_nodes)
    split = int(0.8 * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[indices[:split]] = True
    val_mask[indices[split:]] = True

    in_channels = subgraph.embeddings.shape[1]
    out_channels = len(hypergraph.full_label_map)
    model = arXivHGNN(in_channels, hidden_channels=128, out_channels=out_channels).to(device)

    print("\nTraining on subgraph...")
    model = train(model, subgraph, train_mask, val_mask, epochs=100, device=device)

    evaluate_per_label(model, subgraph, val_mask, hypergraph.full_label_map, device=device)

    print("\n✅ Training complete.")

if __name__ == "__main__":
    main()