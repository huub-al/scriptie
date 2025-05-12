# train_allset.py

import torch
import numpy as np
from topomodelx.nn.hypergraph.allset_transformer import AllSetTransformer
import torch.backends

import sys
sys.path.append("/Users/huubal/scriptie/data")
from paperNodes_graph import HypergraphDataset

class Network(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, task_level="node", **kwargs):
        super().__init__()
        self.base_model = AllSetTransformer(in_channels, hidden_channels, **kwargs)
        self.linear = torch.nn.Linear(hidden_channels, out_channels)
        self.out_pool = task_level == "graph"

    def forward(self, x_0, incidence_1):
        x_0, _ = self.base_model(x_0, incidence_1)
        x = torch.max(x_0, dim=0)[0] if self.out_pool else x_0
        return self.linear(x)

def acc_fn(y_true, y_pred):
    return (y_true == y_pred).float().mean()

def main():
    device = torch.device("cpu" if torch.backends.mps.is_available() else "cpu")
    dataset = torch.load("data/arxiv-data/cs_2000.pt")
    x, y, incidence = dataset.x.to(device), dataset.y.to(device), dataset.incidence.to(device)
    train_mask = dataset.train_mask.to(device)
    val_mask = dataset.val_mask.to(device)
    test_mask = dataset.test_mask.to(device)

    in_channels = x.shape[1]
    out_channels = len(torch.unique(y))
    hidden_channels = 256 

    model = Network(in_channels, hidden_channels, out_channels, task_level="node").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    num_epochs = 200
    for epoch in range(1, num_epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(x, incidence)
        loss = loss_fn(logits[train_mask], y[train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(x, incidence)
            train_acc = acc_fn(y[train_mask], logits[train_mask].argmax(dim=1))
            val_acc = acc_fn(y[val_mask], logits[val_mask].argmax(dim=1))
            test_acc = acc_fn(y[test_mask], logits[test_mask].argmax(dim=1))

        print(
            f"Epoch {epoch:02d} | Loss: {loss.item():.4f} | "
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}"
        )

if __name__ == "__main__":
    main()
