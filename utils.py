"""
Huub Al

utils.py
"""

import torch


def append_sparse_column(sparse_incidence, new_column, num_nodes):
    """
    sparse_incidence: existing sparse tensor [N, E]
    new_column: sparse tensor [N, 1] (new edge)
    Returns: [N, E+1] sparse tensor
    """
    indices1 = sparse_incidence._indices()
    values1 = sparse_incidence._values()
    
    indices2 = new_column._indices()
    values2 = new_column._values()
    
    # Shift new column index from 0 to existing num_edges
    indices2[1] += sparse_incidence.size(1)
    
    new_indices = torch.cat([indices1, indices2], dim=1)
    new_values = torch.cat([values1, values2])
    
    return torch.sparse_coo_tensor(
        new_indices,
        new_values,
        size=(num_nodes, sparse_incidence.size(1) + 1)
    )
