"""
hypergraph_dummy.py
Huub Al 

This script creates a dummy hypergraph with random data matching the arXiv structure for testing purposes.
Each node represents either an author or a paper, and each hyperedge represents a paper's authorship.
"""
import torch
from torch_geometric.data import Data
import numpy as np
from torch_sparse import coalesce



class DummyHyperGraph:
    def __init__(self, num_authors=1000, num_papers=500, feature_dim=768):
        """
        Create a dummy hypergraph for testing
        
        Args:
            num_authors (int): Number of authors in the graph
            num_papers (int): Number of papers in the graph
            feature_dim (int): Dimension of node features
        """
        self.num_authors = num_authors
        self.num_papers = num_papers
        self.feature_dim = feature_dim
        
        # Create node features
        self.author_features = torch.randn(num_authors, feature_dim)
        self.paper_features = torch.randn(num_papers, feature_dim)
        
        # Create random authorship (each paper has 1-5 authors)
        self.author_paper_map = {}  # paper_idx -> list of author indices
        for paper_idx in range(num_papers):
            num_authors = np.random.randint(1, 6)  # Random number of authors (1-5)
            authors = np.random.choice(self.num_authors, num_authors, replace=False)
            self.author_paper_map[paper_idx] = authors.tolist()
    
    def get_data(self):
        """
        Get the data in PyTorch Geometric format
        
        Returns:
            Data: PyTorch Geometric data object
        """
        # Create node features
        x = torch.cat([self.author_features, self.paper_features], dim=0)
        
        # Create edge indices (author-paper connections)
        edge_index = []
        for paper_idx, authors in self.author_paper_map.items():
            paper_idx_with_offset = paper_idx + self.num_authors  # Add offset for papers
            for author_idx in authors:
                # Add both directions (V->E and E->V)
                edge_index.append([author_idx, paper_idx_with_offset])
                edge_index.append([paper_idx_with_offset, author_idx])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_index, _ = coalesce(edge_index, None, self.num_authors + self.num_papers, self.num_authors + self.num_papers)
        
        # Create number of nodes and hyperedges
        n_x = torch.tensor([self.num_authors, self.num_papers])
        num_hyperedges = torch.tensor([self.num_papers])  # Each paper is a hyperedge
        
        # Create data object
        data = Data(
            x=x,
            edge_index=edge_index,
            n_x=n_x,
            num_hyperedges=num_hyperedges
        )
        
        return data
    
    def get_author_idx(self, author_name):
        """
        Get index of an author by name
        
        Args:
            author_name (str): Name of the author (e.g., 'author_0')
            
        Returns:
            int: Index of the author, or None if not found
        """
        try:
            return int(author_name.split('_')[1])
        except:
            return None
    
    def get_paper_idx(self, paper_name):
        """
        Get index of a paper by name
        
        Args:
            paper_name (str): Name of the paper (e.g., 'paper_0')
            
        Returns:
            int: Index of the paper, or None if not found
        """
        try:
            return int(paper_name.split('_')[1])
        except:
            return None
    
    def get_author_name(self, idx):
        """
        Get name of an author by index
        
        Args:
            idx (int): Index of the author
            
        Returns:
            str: Name of the author
        """
        return f'author_{idx}'
    
    def get_paper_name(self, idx):
        """
        Get name of a paper by index
        
        Args:
            idx (int): Index of the paper
            
        Returns:
            str: Name of the paper
        """
        return f'paper_{idx}' 