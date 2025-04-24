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
from scipy import sparse
import re
from transformers import AutoTokenizer, AutoModel


class ArxivHyperGraph:
    """
    Constructs main arXiv Hypergraph
    """
    def __init__(self, input_file):
        """
        Loads existing arXiv structure or creates anew.
        
        Args:
            input_file (str): Path to the JSON.GZ file containing arXiv data
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
        self.model.eval()  # Set to evaluation mode
        
        # Initialize data structures
        self.edge_dict = {}  # dict with keys as edge_ids and values as paper data
        self.author_dict = {}  # dict with keys as author_ids and values as list of paper IDs
        self.paper_to_authors = {}  # dict mapping paper IDs to their authors
        
        # Load and process the JSON.GZ file
        with gzip.open(input_file, 'rt', encoding='utf-8') as f:
            for line in f:
                paper = json.loads(line)
                paper_id = paper['id']
                
                # Generate SciBERT embedding for abstract
                abstract = paper['abstract']
                # Tokenize and get model output
                with torch.no_grad():
                    inputs = self.tokenizer(abstract, return_tensors="pt", truncation=True, max_length=512)
                    outputs = self.model(**inputs)
                    # Use [CLS] token embedding as paper representation
                    embedding = outputs.last_hidden_state[0, 0, :].numpy()
                
                # Store paper data and embedding in edge_dict
                self.edge_dict[paper_id] = {
                    'data': paper,
                    'embedding': embedding
                }
                
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
                
                self.paper_to_authors[paper_id] = filtered_authors
                
                # Add paper to each author's list
                for author in filtered_authors:
                    if author not in self.author_dict:
                        self.author_dict[author] = []
                    self.author_dict[author].append(paper_id)
        
        # Create mapping from IDs to matrix indices
        self.paper_to_idx = {pid: idx for idx, pid in enumerate(self.edge_dict.keys())}
        self.author_to_idx = {aid: idx for idx, aid in enumerate(self.author_dict.keys())}
        
        # Create sparse adjacency matrix
        num_papers = len(self.edge_dict)
        num_authors = len(self.author_dict)
        
        # Create lists for sparse matrix construction
        rows, cols, data = [], [], []
        
        # Fill adjacency matrix data
        for paper_id, authors in self.paper_to_authors.items():
            paper_idx = self.paper_to_idx[paper_id]
            for author in authors:
                author_idx = self.author_to_idx[author]
                rows.append(paper_idx)
                cols.append(author_idx)
                data.append(1.0)
        
        # Create sparse matrix
        self.A = sparse.csr_matrix((data, (rows, cols)), shape=(num_papers, num_authors), dtype=np.float32)
        
        # Save to pickle file
        print(f"Saving to pickle file: {pickle_file}")
        with open(pickle_file, 'wb') as f:
            pickle.dump(self, f)

class ArxivSubGraph(ArxivHyperGraph):
    """
    Constructs subset of arXiv graph.
    """
    def __init__(self, input_file, dropout=0.1):
        """
        constructs subset of graph with some
        edges dropped out.
        
        Args:
            input_file (str): Path to the JSON.GZ file containing arXiv data
            dropout (float): Probability of dropping an edge (default: 0.1)
        """
        super().__init__(input_file)
        self.sub_A = 0 # new strictly smaller adjacency matrix.
        self.edges_left = [] # list with keys of edges that are left.
        self.node_features = {} # dict with aggregated embeddings, 
        # should pay attention to information leakage. 

    def generate_edge(self):
        """
        generates a fake hyper edge.
        """



"""
- wat referenties uit de papers bekijken
- human in the loop / interactive learning 
- plaatjes/blokschema maken in powerpoint draw.io
"""