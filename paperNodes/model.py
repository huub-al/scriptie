"""
Huub Al

Python script that initializes a HGNN model.
"""
import torch
import numpy as np
from topomodelx.nn.hypergraph.allset_transformer import AllSetTransformer
import torch.backends


class arXivHGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, task_level="node", **kwargs):
        super().__init__()
        self.base_model = AllSetTransformer(in_channels, hidden_channels, **kwargs)
        self.linear = torch.nn.Linear(hidden_channels, out_channels)
        self.out_pool = task_level == "graph"

    def forward(self, x_0, incidence_1):
        x_0, _ = self.base_model(x_0, incidence_1)
        x = torch.max(x_0, dim=0)[0] if self.out_pool else x_0
        return self.linear(x)