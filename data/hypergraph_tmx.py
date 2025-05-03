"""
hypergraph.py
Huub Al 

This script contains the hypergraph class and 
graph subset class for training and inference of
the AllSetTransformer via https://github.com/jianhao2016/AllSet
"""
import numpy as np
import torch
import json
import gzip
import pickle
import os
import re
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


class ArxivHyperGraph:
    """
    Constructs main arXiv Hypergraph
    """
    def __init__(self, input_file, batch_size=32):
        """
        Loads existing arXiv structure or creates anew.
        
        Args:
            input_file (str): Path to the JSON.GZ file containing arXiv data
            batch_size (int): Number of papers to process at once (default: 32)
        """
        # Check if pickle file exists
        pickle_file = input_file.replace('.json.gz', '.pkl')
        if os.path.exists(pickle_file):
            print(f"Loading from pickle file: {pickle_file}")
            with open(pickle_file, 'rb') as f:
                loaded_graph = pickle.load(f)
                self.__dict__.update(loaded_graph.__dict__)
            return

        # Initialize SciBERT model and tokenizer
        print("Loading SciBERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self.model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        
        # Move model to GPU if available
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Initialize data structures
        self.edge_dict = {}  # dict with keys as edge_ids and values as paper data
        self.author_dict = {}  # dict with keys as author_ids and values as list of paper IDs
        self.paper_to_authors = {}  # dict mapping paper IDs to their authors
        self.max_authors = 0  # maximum number of authors in any paper
        
        # Load and process the JSON.GZ file
        print("Loading papers and generating embeddings...")
        papers = []
        with gzip.open(input_file, 'rt', encoding='utf-8') as f:
            for line in f:
                papers.append(json.loads(line))
        
        # Process papers in batches
        for i in tqdm(range(0, len(papers), batch_size)):
            batch = papers[i:i + batch_size]
            
            # Process each paper in the batch
            for paper in batch:
                paper_id = paper['id']
                
                # Process authors - filter out institutional entries
                authors = paper['authors'].split(', ')
                # Filter out entries that look like institutions or countries
                filtered_authors = []
                for author in authors:
                    # Skip if author appears to be an institution, country, or 'et al'
                    if (not (re.match(r'^[A-Za-z\s]+\)?$', author) and len(author.split()) <= 2) and
                        not any(indicator in author.lower() for indicator in ['university', 'institute', 'center', 'lab', 'department']) and
                        author.lower() not in ['usa', 'uk', 'germany', 'france', 'italy', 'russia', 'china', 'japan'] and
                        author.lower() not in ['et al', 'et al.', 'et.al', 'et.al.']):
                        filtered_authors.append(author)
                
                # Update max_authors if this paper has more authors
                self.max_authors = max(self.max_authors, len(filtered_authors))
                
                self.paper_to_authors[paper_id] = filtered_authors
                
                # Add paper to each author's list
                for author in filtered_authors:
                    if author not in self.author_dict:
                        self.author_dict[author] = []
                    self.author_dict[author].append(paper_id)
            
            # Generate embeddings for the batch
            abstracts = [paper['abstract'] for paper in batch]
            with torch.no_grad():
                inputs = self.tokenizer(abstracts, return_tensors="pt", truncation=True, max_length=512, padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # Store paper data and embeddings
            for paper, embedding in zip(batch, embeddings):
                paper_id = paper['id']
                self.edge_dict[paper_id] = embedding

        # Calculate mean author count
        author_counts = [len(authors) for authors in self.paper_to_authors.values() if len(authors) > 0]
        self.poisson_lambda = np.mean(author_counts)
        
        # Create mapping from paper IDs to matrix indices
        self.paper_to_idx = {pid: idx for idx, pid in enumerate(self.edge_dict.keys())}
        self.idx_to_paper =  {idx: pid for idx, pid in enumerate(self.edge_dict.keys())}
        # Create mapping from authors to matrix indices
        self.author_to_idx = {aid: idx for idx, aid in enumerate(self.author_dict.keys())}
        self.idx_to_author = {idx: aid for idx, aid in enumerate(self.author_dict.keys())}
        
        # Save to pickle file
        print(f"Saving to pickle file: {pickle_file}")
        with open(pickle_file, 'wb') as f:
            pickle.dump(self, f)

    def gen_subgraph(self, dropout=0.1):
        """
        Generates a subgraph by randomly dropping entire edges (papers) while preserving all authors.
        
        Args:
            dropout (float): Probability of dropping an edge (default: 0.1)
            
        Returns:
            ArxivSubGraph: A new subgraph instance with dropped edges
        """
        # Get all paper ids 
        all_papers = list(self.paper_to_idx.keys())
        
        # First pass: randomly select papers to drop using vectorized operations
        random_mask = np.random.random(len(all_papers)) < dropout
        papers_to_drop = set(np.array(all_papers)[random_mask])
        papers_to_keep = set(np.array(all_papers)[~random_mask])

        # sanity check
        assert len(papers_to_drop) + len(papers_to_keep) == len(all_papers)
        
        # Check which authors would be left without any papers
        authors_without_papers = set()
        for author, papers in self.author_dict.items():
            if all(paper in papers_to_drop for paper in papers):
                authors_without_papers.add(author)
        
        # Second pass: ensure all authors have at least one paper
        for author in authors_without_papers:
            # Get all papers for this author
            author_papers = set(self.author_dict[author])
            # Move one random paper from drop to keep
            # from previous iterations an author might have already
            # been assigned a paper to keep, so we need to check for that
            available_papers = author_papers.intersection(papers_to_drop)
            if len(available_papers) == len(author_papers):
                paper_to_keep = np.random.choice(list(available_papers))
                papers_to_drop.remove(paper_to_keep)
                papers_to_keep.add(paper_to_keep)

        # construct sparse incidence matrix
        rows, cols = [], []

        for paper_id in papers_to_keep:
            paper_idx = self.paper_to_idx[paper_id]
            for author in self.paper_to_authors[paper_id]:
                author_idx = self.author_to_idx[author]
                rows.append(author_idx)
                cols.append(paper_idx)

        indices = torch.tensor([rows, cols], dtype=torch.long)
        values = torch.ones(len(rows), dtype=torch.float32)
        sub_H = torch.sparse_coo_tensor(indices, values, size=(len(self.author_to_idx), len(self.paper_to_idx)))

        # Compute node features
        # Get embedding dimension from first paper
        first_paper_id = list(self.edge_dict.keys())[0]
        embedding_dim = self.edge_dict[first_paper_id].shape[0]
        
        # Initialize node features tensor
        num_authors = len(self.author_to_idx)
        node_features = torch.zeros((num_authors, embedding_dim), dtype=torch.float32)
        
        # For each author, compute mean of their paper embeddings in the subgraph
        for author, author_idx in self.author_to_idx.items():
            # Get papers this author contributed to that are in the subgraph
            author_papers = [pid for pid in self.author_dict[author] if pid in papers_to_keep]
            
            if author_papers:
                # Get embeddings for these papers
                paper_embeddings = [self.edge_dict[pid] for pid in author_papers]
                # Compute mean embedding
                mean_embedding = np.mean(paper_embeddings, axis=0)
                node_features[author_idx] = torch.tensor(mean_embedding, dtype=torch.float32)
            else:
                raise ValueError(f"Author {author} has no papers in the subgraph")
        
        # Create and return subgraph
        subgraph = ArxivSubGraph(self)
        subgraph.sub_H = sub_H
        subgraph.edges_left = list(papers_to_keep)
        subgraph.dropped_edges = list(papers_to_drop)
        subgraph.node_features = node_features
        
        return subgraph


class ArxivSubGraph(ArxivHyperGraph):
    """
    Constructs subset of arXiv graph.
    """
    def __init__(self, parent_graph):
        """
        Initializes a subgraph from a parent graph.
        
        Args:
            parent_graph (ArxivHyperGraph): The parent graph to create a subgraph from
        """
        # Copy necessary attributes from parent graph
        self.edge_dict = parent_graph.edge_dict
        self.author_dict = parent_graph.author_dict
        self.paper_to_authors = parent_graph.paper_to_authors
        self.paper_to_idx = parent_graph.paper_to_idx
        self.author_to_idx = parent_graph.author_to_idx
        self.tokenizer = parent_graph.tokenizer
        self.max_authors = parent_graph.max_authors
        self.model = parent_graph.model
        self.device = parent_graph.device
        self.poisson_lambda = parent_graph.poisson_lambda
        
        # Initialize subgraph specific attributes
        self.sub_H = None
        self.edges_left = []
        self.dropped_edges = []
        self.node_features = None

    def generate_plausible_fake_edge(self):
        """
        Generates a fake hyperedge by combining random authors.
        Number of authors is poisson distributed and scaled to range [1, max_authors].
        Uses an abstract from a dropped paper that none of the selected authors have written.
        
        Returns:
            tuple: A tuple containing (paper_embedding, author_indices) where both are torch tensors
        """
        # Get all available authors
        available_authors = self.sub_H.shape[0]

        # Generate number of authors from Poisson distribution
        num_authors = np.random.poisson(self.poisson_lambda)
        num_authors = int(np.clip(num_authors, 1, self.max_authors))  # clamp to avoid 0 or excessive authors
            
        # Randomly select authors
        selected_authors = torch.tensor(np.random.choice(available_authors, num_authors, replace=False), dtype=torch.long)
        
        # Find a dropped paper that none of the selected authors have written
        available_dropped_papers = []
        for paper_id in self.dropped_edges:
            paper_authors = set(self.paper_to_authors[paper_id])
            if not any(author in paper_authors for author in selected_authors):
                available_dropped_papers.append(paper_id)
        
        if not available_dropped_papers:
            # If no suitable paper found, just use the first dropped paper
            paper_id = self.dropped_edges[0]
        else:
            paper_id = np.random.choice(available_dropped_papers)
        
        paper_embedding = torch.tensor(self.edge_dict[paper_id], dtype=torch.float)
        return paper_embedding, selected_authors

    def generate_fake_edge(self):
        """
        Generates a fake hyperedge by combining random authors.
        Number of authors is poisson distributed and scaled to range [1, max_authors].
        Uses a random 768-dimensional embedding as paper embedding.
        
        Returns:
            tuple: A tuple containing (paper_embedding, author_indices) where both are torch tensors
        """
        available_authors = self.sub_H.shape[0]
        num_authors = np.random.poisson(self.poisson_lambda)
        num_authors = int(np.clip(num_authors, 1, self.max_authors))  # clamp to avoid 0 or excessive authors
        selected_authors = torch.tensor(np.random.choice(available_authors, num_authors, replace=False), dtype=torch.long)
        paper_embedding = torch.randn(768, dtype=torch.float)
        return paper_embedding, selected_authors

    def generate_real_edge(self):
        """
        Returns all real edges (papers) that are not in the current subgraph.
        Only includes papers that have valid authors.
        Returns a list of tuples, each containing the paper embedding and the indices of its authors.
        
        Returns:
            list: A list of tuples, each containing (paper_embedding, author_indices) where both are torch tensors
        """
        real_edges = []
        for paper_id in self.dropped_edges:
            # Check if paper has authors
            if paper_id in self.paper_to_authors and len(self.paper_to_authors[paper_id]) > 0:
                # Get the paper's embedding
                paper_embedding = torch.tensor(self.edge_dict[paper_id], dtype=torch.float)
                
                # Get the authors of this paper and convert to indices
                paper_authors = self.paper_to_authors[paper_id]
                author_indices = torch.tensor([self.author_to_idx[author] for author in paper_authors], dtype=torch.long)
                
                real_edges.append((paper_embedding, author_indices))
            
        return real_edges


"""
- wat referenties uit de papers bekijken
- human in the loop / interactive learning 
- plaatjes/blokschema maken in powerpoint draw.io
"""