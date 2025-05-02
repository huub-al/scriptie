"""
hypergraph_v2.py
Huub Al 

This script contains the hypergraph class for SetGNN compatibility.
It converts the arXiv dataset into a format suitable for SetGNN training.
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
from tqdm import tqdm
from torch_geometric.data import Data
from torch_sparse import coalesce


class ArxivHyperGraphSetGNN:
    """
    Constructs arXiv Hypergraph in SetGNN compatible format
    """
    def __init__(self, input_file, batch_size=32):
        """
        Loads existing arXiv structure or creates anew.
        
        Args:
            input_file (str): Path to the JSON.GZ file containing arXiv data
            batch_size (int): Number of papers to process at once (default: 32)
        """
        # Check if pickle file exists
        pickle_file = input_file.replace('.json.gz', '_setgnn.pkl')
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Initialize data structures
        self.edge_dict = {}  # dict with keys as edge_ids and values as paper data
        self.author_dict = {}  # dict with keys as author_ids and values as list of paper IDs
        self.paper_to_authors = {}  # dict mapping paper IDs to their authors
        
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
                # Take only the first category
                category = paper['categories'][0]
                self.edge_dict[paper_id] = {
                    'data': paper,
                    'embedding': embedding,
                    'category': category
                }
        
        # Create mapping from IDs to matrix indices
        self.paper_to_idx = {pid: idx for idx, pid in enumerate(self.edge_dict.keys())}
        self.author_to_idx = {aid: idx for idx, aid in enumerate(self.author_dict.keys())}
        
        # Create edge index for SetGNN format
        num_nodes = len(self.author_dict)  # Authors are nodes
        num_hyperedges = len(self.edge_dict)  # Papers are hyperedges
        
        # Create lists for edge index construction
        node_list = []
        edge_list = []
        
        # Fill edge index data
        for paper_id, authors in self.paper_to_authors.items():
            paper_idx = self.paper_to_idx[paper_id] + num_nodes  # Hyperedge indices start after nodes
            for author in authors:
                if author in self.author_to_idx:  # Skip authors that were filtered out
                    author_idx = self.author_to_idx[author]
                    node_list.append(author_idx)
                    edge_list.append(paper_idx)
        
        # Create edge index tensor
        edge_index = np.array([node_list + edge_list, edge_list + node_list], dtype=np.int64)
        edge_index = torch.LongTensor(edge_index)
        
        # Create hyperedge features from paper embeddings
        hyperedge_features = torch.zeros(num_hyperedges, 768)  # 768 is the SciBERT embedding dimension
        for paper_id, paper_data in self.edge_dict.items():
            paper_idx = self.paper_to_idx[paper_id]
            hyperedge_features[paper_idx] = torch.tensor(paper_data['embedding'])
        
        # Create edge attributes by repeating hyperedge features for each edge
        edge_attr = torch.zeros(edge_index.size(1), 768)
        for i, (src, dst) in enumerate(edge_index.t()):
            if dst >= num_nodes:  # If this is a hyperedge
                edge_attr[i] = hyperedge_features[dst - num_nodes]
            else:  # If this is a node
                edge_attr[i] = hyperedge_features[src - num_nodes]
        
        # Coalesce edge index and edge attributes to remove duplicates and sort
        total_num_node_id_he_id = edge_index.max() + 1
        edge_index, edge_attr = coalesce(edge_index, edge_attr, total_num_node_id_he_id, total_num_node_id_he_id)
        
        # Create node features (author embeddings)
        # For now, we'll use random features since we don't have author embeddings
        # In a real implementation, you might want to use author-specific features
        node_features = torch.randn(num_nodes, 768)  # 768 is the SciBERT embedding dimension
        
        # Create labels (paper categories)
        # For now, we'll use random labels since we don't have a specific task
        # In a real implementation, you would use actual labels based on your task
        num_classes = len(set(paper['category'] for paper in self.edge_dict.values()))
        labels = torch.randint(0, num_classes, (num_nodes,))
        
        # Create PyTorch Geometric Data object
        self.data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,  # Add hyperedge features
            y=labels,
            n_x=torch.tensor([num_nodes]),
            num_hyperedges=torch.tensor([num_hyperedges])
        )
        
        # Save to pickle file
        print(f"Saving to pickle file: {pickle_file}")
        with open(pickle_file, 'wb') as f:
            pickle.dump(self, f)
    
    def get_data(self):
        """
        Returns the PyTorch Geometric Data object
        """
        return self.data
