"""
ArXiv Hypergraph Data Classes

This module provides classes for constructing and manipulating hypergraphs
from arXiv paper data. It includes functionality for embedding papers,
tracking author relationships, and dynamically updating graph structure
with real, plausible, and fake papers.
"""

import os
import re
import gzip
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel


class arXivHyperGraph:
    """
    A hypergraph representation of arXiv papers and their author relationships.
    
    This class loads arXiv paper data, computes embeddings for paper abstracts,
    and maintains the relationships between papers and authors. It serves as the
    base graph from which subgraphs can be constructed.
    
    Attributes:
        data_path (str): Path to the arXiv data file (gzipped JSON)
        model_name (str): Name of the pretrained model for embeddings
        cache_path (str): Path to save/load cached graph data
        device (torch.device): Device for computation (CPU or CUDA)
        x (torch.Tensor): Paper embeddings matrix
        y (torch.Tensor): Paper labels tensor
        paper_ids (list): List of arXiv paper IDs
        node_to_authors (dict): Mapping from node indices to author lists
        author_pool (list): List of all unique authors
        author_mean (float): Average number of authors per paper
        paper_id_to_idx (dict): Mapping from paper IDs to node indices
        label_map (dict): Mapping from category names to label indices
        synthetic_labels (dict): Mapping from synthetic label names to indices
        full_label_map (dict): Combined mapping of real and synthetic labels
    """
    
    def __init__(self, data_path="arxiv-data/subset_cs_2000.json.gz",
                 model_name="allenai/scibert_scivocab_uncased",
                 cache_path="arxiv_hypergraph.pt"):
        """
        Initialize the arXiv hypergraph.
        
        Args:
            data_path (str): Path to the arXiv data file (gzipped JSON)
            model_name (str): Name of the pretrained model for embeddings
            cache_path (str): Path to save/load cached graph data
        """
        self.data_path = data_path
        self.model_name = model_name
        self.cache_path = cache_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if os.path.exists(self.cache_path):
            print(f"Loading cached graph from {self.cache_path}")
            self._load_cache()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self._build()
            self._save_cache()

    def _build(self):
        """
        Build the hypergraph from arXiv data.
        
        This method loads papers, extracts abstracts and authors, computes
        embeddings, and constructs the necessary data structures for the graph.
        """
        papers = self._load_papers()
        abstracts, paper_ids, labels, paper_to_authors = [], [], [], {}

        # Create base label mapping from observed category labels
        all_categories = sorted({p["categories"][0] for p in papers if p.get("abstract")})
        self.label_map = {cat: i for i, cat in enumerate(all_categories)}
        
        # Define synthetic labels with fixed indices to avoid confusion
        self.synthetic_labels = {
            "not a likely collaboration": len(self.label_map),
            "not scientific work": len(self.label_map) + 1
        }
        self.full_label_map = {**self.label_map, **self.synthetic_labels}
        
        # Create reverse mapping for debugging and validation
        self.idx_to_label = {idx: label for label, idx in self.full_label_map.items()}

        for paper in papers:
            if not paper.get("abstract"):
                continue
            authors = self._filter_authors(paper["authors"])
            if not authors:
                continue
            abstracts.append(paper["abstract"])
            paper_ids.append(paper["id"])
            labels.append(self.label_map[paper["categories"][0]])
            paper_to_authors[paper["id"]] = authors

        self.x = self._get_embeddings(abstracts)
        self.y = torch.tensor(labels, dtype=torch.long)
        self.paper_ids = paper_ids
        self.node_to_authors = {i: paper_to_authors[pid] for i, pid in enumerate(paper_ids)}
        self.author_pool = list({a for authors in paper_to_authors.values() for a in authors})
        self.author_mean = np.mean([len(authors) for authors in paper_to_authors.values()])
        self.paper_id_to_idx = {pid: idx for idx, pid in enumerate(paper_ids)}

    def _print_summary(self):
        """
        Print a summary of the hypergraph statistics.
        """
        print("\n--- Hypergraph Summary ---")
        print(f"Total papers: {len(self.paper_ids)}")
        print(f"Embedding shape: {self.x.shape}")
        print(f"Number of classes: {len(self.label_map)}")
        print(f"Average authors per paper: {self.author_mean:.2f}")

    def _load_cache(self):
        """
        Load the hypergraph from a cached file.
        """
        state = torch.load(self.cache_path, weights_only=False)
        self.x = state["x"]
        self.y = state["y"]
        self.paper_ids = state["paper_ids"]
        self.node_to_authors = state["node_to_authors"]
        self.author_mean = state["author_mean"]
        self.author_pool = state["author_pool"]
        self.paper_id_to_idx = state["paper_id_to_idx"]
        self.label_map = state["label_map"]
        self.synthetic_labels = state["synthetic_labels"]
        self.full_label_map = {**self.label_map, **self.synthetic_labels}
        
        # Create reverse mapping for debugging and validation
        self.idx_to_label = {idx: label for label, idx in self.full_label_map.items()}

    def _save_cache(self):
        """
        Save the hypergraph to a cache file for faster loading.
        """
        torch.save({
            "x": self.x,
            "y": self.y,
            "paper_ids": self.paper_ids,
            "node_to_authors": self.node_to_authors,
            "author_mean": self.author_mean,
            "author_pool": self.author_pool,
            "paper_id_to_idx": self.paper_id_to_idx,
            "label_map": self.label_map,
            "synthetic_labels": self.synthetic_labels
        }, self.cache_path)

    def _load_papers(self):
        """
        Load papers from the gzipped JSON file.
        
        Returns:
            list: List of paper dictionaries
        """
        with gzip.open(self.data_path, "rt", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def _filter_authors(self, author_string):
        """
        Filter and clean author names from the author string.
        
        Args:
            author_string (str): Raw author string from the paper data
            
        Returns:
            list: List of cleaned author names
        """
        authors = re.split(r',\s*|\s+and\s+', author_string)
        filtered = []
        for author in authors:
            a = author.strip().lower()
            if any(word in a for word in ['university', 'institute', 'lab', 'center', 'department']):
                continue
            if a in ['usa', 'uk', 'china', 'japan', 'germany', 'et al']:
                continue
            if re.match(r'^[a-zA-Z.\- ]+$', author) and len(author.split()) <= 3:
                filtered.append(author.strip())
        return filtered

    def _get_embeddings(self, texts):
        """
        Compute embeddings for paper abstracts using the pretrained model.
        
        Args:
            texts (list): List of paper abstract texts
            
        Returns:
            torch.Tensor: Tensor of paper embeddings
        """
        embeddings = []
        for text in tqdm(texts, desc="Embedding abstracts"):
            inputs = self.tokenizer(text, truncation=True, padding=True,
                                    max_length=512, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()
            embeddings.append(cls)
        return torch.stack(embeddings)

    def construct_subgraph(self, dropout=0.0):
        """
        Construct a subgraph from the full hypergraph.
        
        Args:
            dropout (float): Fraction of papers to exclude from the subgraph
            
        Returns:
            arXivSubGraph: A subgraph object containing the selected papers
        """
        mask = np.random.rand(len(self.paper_ids)) > dropout
        indices = np.where(mask)[0]
        return arXivSubGraph(self, indices)


class arXivSubGraph:
    """
    A subgraph of the arXiv hypergraph that can be dynamically modified.
    
    This class maintains a subset of papers from the full hypergraph and
    provides methods to add real, plausible, and fake papers, as well as
    to remove fake papers. It maintains the incidence matrix representing
    the author-paper relationships.
    
    Attributes:
        hypergraph (arXivHyperGraph): Reference to the parent hypergraph
        indices (list): List of indices in the parent graph
        fake_papers (list): List of indices of fake papers in the subgraph
        plausible_papers (list): List of indices of plausible papers in the subgraph
        synthetic_papers (list): List of indices of all papers with synthetic labels
        embeddings (torch.Tensor): Paper embeddings matrix for the subgraph
        labels (torch.Tensor): Paper labels tensor for the subgraph
        node_to_authors (dict): Mapping from node indices to author lists
        incidence (torch.Tensor): Sparse incidence matrix of author-paper relationships
    """
    
    def __init__(self, hypergraph: arXivHyperGraph, indices):
        """
        Initialize the subgraph from a subset of the full hypergraph.
        
        Args:
            hypergraph (arXivHyperGraph): The parent hypergraph
            indices (list or array): Indices of papers to include in the subgraph
        """
        self.hypergraph = hypergraph
        self.indices = list(indices)
        self.fake_papers = []
        self.plausible_papers = []
        self.synthetic_papers = []  # Track all papers with synthetic labels

        # Extract relevant data for the subgraph
        self.embeddings = hypergraph.x[indices]
        self.labels = hypergraph.y[indices].clone()
        self.node_to_authors = {i: hypergraph.node_to_authors[idx] for i, idx in enumerate(indices)}

        # Build the initial incidence matrix
        self._rebuild_incidence()

    def _rebuild_incidence(self):
        """
        Rebuild the incidence matrix for the current state of the subgraph.
        
        This creates a sparse matrix where rows represent papers and columns
        represent authors. A value of 1 indicates that an author is associated
        with a paper.
        """
        # Create mapping from authors to papers
        author_to_papers = defaultdict(list)
        for idx, authors in self.node_to_authors.items():
            for author in authors:
                author_to_papers[author].append(idx)

        # Create the incidence matrix
        num_nodes = len(self.node_to_authors)
        num_edges = len(author_to_papers)
        incidence = torch.zeros((num_nodes, num_edges), dtype=torch.float32)

        # Fill the incidence matrix
        for j, (author, papers) in enumerate(author_to_papers.items()):
            for i in papers:
                incidence[i, j] = 1
                
        # Convert to sparse format for efficiency
        self.incidence = incidence.to_sparse_coo()

    def add_papers(self, n, fake=0.3, plausible=0.3):
        """
        Add papers to the subgraph with specified proportions of real, fake, and plausible papers.
        
        Args:
            n (int): Total number of papers to add
            fake (float): Fraction of papers that should be fake
            plausible (float): Fraction of papers that should be plausible
            
        Note:
            - Real papers are actual papers from the parent graph not yet in the subgraph
            - Fake papers have random embeddings and are labeled as "not scientific work"
            - Plausible papers use embeddings from existing papers but with new author sets,
              and are labeled as "not a likely collaboration"
        """
        n_fake = int(n * fake)
        n_plausible = int(n * plausible)
        n_real = n - n_fake - n_plausible
        mean_authors = self.hypergraph.author_mean

        new_embeddings, new_labels, new_authors = [], [], []

        # Add real papers (with original labels)
        current_ids = {self.hypergraph.paper_ids[i] for i in self.indices}
        remaining = [i for i, pid in enumerate(self.hypergraph.paper_ids) if pid not in current_ids]
        selected_real = random.sample(remaining, min(n_real, len(remaining)))

        for idx in selected_real:
            new_embeddings.append(self.hypergraph.x[idx])
            new_labels.append(self.hypergraph.y[idx].item())
            new_authors.append(self.hypergraph.node_to_authors[idx])

        # Get the synthetic label indices directly from the hypergraph
        not_scientific_work_idx = self.hypergraph.synthetic_labels["not scientific work"]
        not_likely_collab_idx = self.hypergraph.synthetic_labels["not a likely collaboration"]

        # Debug: Print synthetic label indices
        # print(f"DEBUG: Synthetic label indices - not_scientific_work: {not_scientific_work_idx}, not_likely_collab: {not_likely_collab_idx}")

        # Add fake papers with synthetic label "not scientific work"
        for _ in range(n_fake):
            emb = torch.randn(self.hypergraph.x.shape[1])
            num_authors = max(1, int(np.random.poisson(mean_authors)))
            authors = random.sample(self.hypergraph.author_pool, num_authors)
            new_embeddings.append(emb)
            new_labels.append(not_scientific_work_idx)
            new_authors.append(authors)
            # Track the index of this fake paper in the subgraph
            paper_idx = len(self.labels) + len(new_labels) - 1
            self.fake_papers.append(paper_idx)
            self.synthetic_papers.append(paper_idx)

        # Add plausible papers with synthetic label "not a likely collaboration"
        for _ in range(n_plausible):
            # Find a paper with authors not already in the graph
            while True:
                idx = random.randint(0, len(self.hypergraph.x) - 1)
                existing_authors = set(self.hypergraph.node_to_authors[idx])
                num_authors = max(1, int(np.random.poisson(mean_authors)))
                candidate_authors = list(set(self.hypergraph.author_pool) - existing_authors)
                if len(candidate_authors) >= num_authors:
                    authors = random.sample(candidate_authors, num_authors)
                    break
            # Use embedding from existing paper but with new author set
            new_embeddings.append(self.hypergraph.x[idx])
            new_labels.append(not_likely_collab_idx)
            new_authors.append(authors)
            # Track the index of this plausible paper in the subgraph
            paper_idx = len(self.labels) + len(new_labels) - 1
            self.plausible_papers.append(paper_idx)
            self.synthetic_papers.append(paper_idx)

        # Combine new papers into the subgraph
        if new_embeddings:
            self.embeddings = torch.cat([self.embeddings, torch.stack(new_embeddings)], dim=0)
            self.labels = torch.cat([self.labels, torch.tensor(new_labels, dtype=torch.long)], dim=0)
            offset = len(self.node_to_authors)
            for i, authors in enumerate(new_authors):
                self.node_to_authors[offset + i] = authors

            # # Debug: Print label distribution after adding papers
            # unique_labels, counts = torch.unique(self.labels, return_counts=True)
            # print("DEBUG: Label distribution after adding papers:")
            # for label, count in zip(unique_labels, counts):
            #     print(f"  Label {label.item()}: {count.item()} papers")

            # Rebuild the incidence matrix to include new papers
            self._rebuild_incidence()

    def remove_fake_papers(self):
        """
        Remove all papers with synthetic labels from the subgraph.
        
        This method removes papers labeled as "not scientific work" and
        "not a likely collaboration", then rebuilds the incidence matrix
        to maintain the correct graph structure.
        """
        if not self.synthetic_papers and len(self.hypergraph.synthetic_labels) == 0:
            return
            
        # Get all synthetic label indices
        synthetic_label_indices = list(self.hypergraph.synthetic_labels.values())
        
        # # Debug: Print synthetic label indices before removal
        # print(f"DEBUG: Removing papers with synthetic labels: {synthetic_label_indices}")
        # print(f"DEBUG: Current labels: {torch.unique(self.labels).tolist()}")
        
        # Create a mask of papers to keep (exclude all papers with synthetic labels)
        keep = []
        for i in range(len(self.labels)):
            label = self.labels[i].item()
            if i not in self.synthetic_papers and label not in synthetic_label_indices:
                keep.append(i)
        
        # # Debug: Print papers to keep and remove
        # print(f"DEBUG: Keeping {len(keep)} papers out of {len(self.labels)}")
        # print(f"DEBUG: Removing {len(self.labels) - len(keep)} papers")
        
        # Update embeddings, labels, and author mappings
        self.embeddings = self.embeddings[keep]
        self.labels = self.labels[keep]
        
        # Rebuild node_to_authors with new indices
        new_node_to_authors = {}
        for i, idx in enumerate(keep):
            new_node_to_authors[i] = self.node_to_authors[idx]
        self.node_to_authors = new_node_to_authors
        
        # Clear all synthetic paper lists
        self.fake_papers.clear()
        self.plausible_papers.clear()
        self.synthetic_papers.clear()
        
        # # Debug: Print final label distribution
        # unique_labels, counts = torch.unique(self.labels, return_counts=True)
        # print("DEBUG: Label distribution after removing synthetic papers:")
        # for label, count in zip(unique_labels, counts):
        #     print(f"  Label {label.item()}: {count.item()} papers")
        
        # Rebuild the incidence matrix
        self._rebuild_incidence()