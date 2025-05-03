"""
Huub Al

This model is a hypergraph model for edge plausibility.

The model is based on the AllSetTransformer model.
"""
import torch
import torch.nn as nn
from topomodelx.nn.hypergraph.allset_transformer import AllSetTransformer

class InteractiveHGNN(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=768):
        super().__init__()
        self.encoder = AllSetTransformer(
            in_channels=in_dim,
            hidden_channels=hidden_dim,
            n_layers=2,
            dropout=0.1,
        )

        # Head A: Hyperedge plausibility (binary classification)
        self.edge_classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )


    def forward(self, x, incidence_matrix):
        """
        we want to provide the linear layer with the
        message passing of the hyperedge and the original
        embedding of the paper.
        x: [num_nodes, in_dim] – input node features
        incidence_matrix: [num_papers, num_authors] – binary matrix
        paper_embedding: [num_papers, in_dim] – embedding of the paper
        """
        _, x_1 = self.encoder(x, incidence_matrix)

        # the last row of the x_1 matrix is the edge
        return self.edge_classifier(x_1[-1])