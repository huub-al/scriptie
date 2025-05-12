# hypergraph_dataset.py

import gzip
import json
import torch
import numpy as np
import os
import re
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class HypergraphDataset:
    def __init__(self, data_path="arxiv-data/subset_cs_200.json.gz", model_name="allenai/scibert_scivocab_uncased", device=None, load_existing=True, dataset_file="arxiv-data/hypergraph_dataset.pt"):
        """
        Initializes the HypergraphDataset object. Loads an existing dataset if available, otherwise builds a new one.

        Args:
            data_path (str): Path to the gzipped JSON dataset.
            model_name (str): Hugging Face model name for embeddings.
            device (torch.device or None): Computation device.
            load_existing (bool): If True, load dataset from disk if available.
            dataset_file (str): File path for saving/loading the dataset.
        """
        self.data_path = data_path
        self.model_name = model_name
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.dataset_file = dataset_file

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        if load_existing and os.path.exists(self.dataset_file):
            self.load()
        else:
            self.build_dataset()
            self.save()


    def load(self):
        """Loads the dataset from a saved file."""
        print("Loading existing dataset...")
        dataset = torch.load(self.dataset_file)
        self.x = dataset.x
        self.y = dataset.y
        self.incidence = dataset.incidence
        self.train_mask = dataset.train_mask
        self.val_mask = dataset.val_mask
        self.test_mask = dataset.test_mask
        self.paper_ids = dataset.paper_ids
        self.node_to_authors = dataset.node_to_authors


    def save(self):
        """Saves the current dataset state to disk."""
        torch.save(self, self.dataset_file)
        print(f"Saved dataset to {self.dataset_file}")


    def build_dataset(self):
        """
        Constructs the hypergraph dataset from scratch.
        Extracts papers, computes embeddings, labels, and builds the incidence matrix.
        Skips papers that have no valid authors after filtering.
        """
        papers = self._load_papers()
        abstracts, labels, paper_ids, paper_to_authors = [], [], [], {}

        for paper in papers:
            if not paper.get("abstract"):
                continue
            filtered_authors = self._filter_authors(paper["authors"])
            if not filtered_authors:
                continue  # Skip papers with no valid authors

            paper_ids.append(paper["id"])
            abstracts.append(paper["abstract"])
            labels.append(paper["categories"][0])
            paper_to_authors[paper["id"]] = filtered_authors

        x = self._get_embeddings(abstracts)
        label_set = sorted(set(labels))
        label_map = {lbl: i for i, lbl in enumerate(label_set)}
        y = torch.tensor([label_map[lbl] for lbl in labels], dtype=torch.long)

        incidence = self._build_incidence_matrix(paper_to_authors)

        indices = np.arange(len(y))
        train_idx, test_val_idx = train_test_split(indices, test_size=0.4, stratify=y, random_state=42)
        val_idx, test_idx = train_test_split(test_val_idx, test_size=0.5, stratify=y[test_val_idx], random_state=42)

        train_mask = torch.zeros(len(y), dtype=torch.bool)
        val_mask = torch.zeros(len(y), dtype=torch.bool)
        test_mask = torch.zeros(len(y), dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        self.x = x
        self.y = y
        self.incidence = incidence
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.paper_ids = paper_ids
        self.node_to_authors = {i: paper_to_authors[pid] for i, pid in enumerate(paper_ids)}


    def _load_papers(self):
        """
        Loads papers from a gzipped JSON file.

        Returns:
            list: List of paper dictionaries.
        """
        papers = []
        with gzip.open(self.data_path, "rt", encoding="utf-8") as f:
            for line in f:
                papers.append(json.loads(line))
        return papers


    def _filter_authors(self, author_string):
        """
        Filters authors to exclude institutions, countries, and 'et al' style entries.

        Args:
            author_string (str): The raw author string from the paper.

        Returns:
            list: A filtered list of author names.
        """
        authors = re.split(r',\s*|\s+and\s+', author_string)
        filtered = []
        for author in authors:
            author_lower = author.strip().lower()
            if any(ex in author_lower for ex in ['university', 'institute', 'center', 'lab', 'department']):
                continue
            if author_lower in ['usa', 'uk', 'germany', 'france', 'italy', 'russia', 'china', 'japan', 'et al', 'et al.', 'et.al', 'et.al.']:
                continue
            if re.match(r'^[A-Za-z\s]+\)?$', author.strip()) and len(author.strip().split()) <= 3:
                filtered.append(author.strip())
        return filtered


    def _get_embeddings(self, texts):
        """
        Computes SciBERT embeddings for a list of abstracts.

        Args:
            texts (list): List of abstract strings.

        Returns:
            torch.Tensor: Tensor of shape [N, D] with embeddings.
        """
        embeddings = []
        for text in tqdm(texts, desc="Embedding abstracts"):
            inputs = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()
            embeddings.append(cls_embedding)
        return torch.stack(embeddings)


    def _build_incidence_matrix(self, paper_to_authors):
        """
        Constructs a sparse incidence matrix from paper-author relationships.

        Args:
            paper_to_authors (dict): Mapping from paper ID to a list of authors.

        Returns:
            torch.Tensor: Sparse COO tensor of shape [num_papers, num_authors].
        """
        author_to_papers = defaultdict(set)
        for paper_id, authors in paper_to_authors.items():
            for author in authors:
                author_to_papers[author].add(paper_id)

        author_hyperedges = list(author_to_papers.values())
        node_ids = {pid: idx for idx, pid in enumerate(paper_to_authors)}
        num_nodes = len(node_ids)
        num_edges = len(author_hyperedges)

        incidence = np.zeros((num_nodes, num_edges), dtype=np.float32)

        for edge_idx, papers in enumerate(author_hyperedges):
            for pid in papers:
                node_idx = node_ids[pid]
                incidence[node_idx, edge_idx] = 1

        return torch.tensor(incidence).to_sparse_coo()


    def add_paper(self, pid, abstract, author_names, label):
        """
        Adds a new paper to the dataset.

        Args:
            pid (str): Paper ID.
            abstract (str): Abstract text.
            author_names (list): List of author names.
            label (str): Category label of the paper.
        """
        if pid in self.paper_ids:
            print(f"Paper {pid} already exists.")
            return

        filtered_authors = self._filter_authors(', '.join(author_names))
        if not filtered_authors:
            print(f"Paper {pid} skipped: no valid authors after filtering.")
            return

        embedding = self._get_embeddings([abstract])[0]
        self.x = torch.cat([self.x, embedding.unsqueeze(0)], dim=0)

        label_idx = self.label_map[label]
        self.y = torch.cat([self.y, torch.tensor([label_idx]).long()], dim=0)

        self.train_mask = torch.cat([self.train_mask, torch.tensor([False])], dim=0)
        self.val_mask = torch.cat([self.val_mask, torch.tensor([False])], dim=0)
        self.test_mask = torch.cat([self.test_mask, torch.tensor([False])], dim=0)

        self.paper_ids.append(pid)
        new_index = len(self.paper_ids) - 1
        self.node_to_authors[new_index] = filtered_authors

        self.incidence = self._build_incidence_matrix(
            {self.paper_ids[i]: self.node_to_authors[i] for i in range(len(self.paper_ids))}
        )
        print(f"Added paper {pid} with label '{label}'.")


    def remove_paper(self, pid):
        """
        Removes a paper from the dataset.

        Args:
            pid (str): Paper ID to remove.
        """
        if pid not in self.paper_ids:
            print(f"Paper {pid} not found.")
            return

        index = self.paper_ids.index(pid)

        self.x = torch.cat([self.x[:index], self.x[index+1:]], dim=0)
        self.y = torch.cat([self.y[:index], self.y[index+1:]], dim=0)
        self.train_mask = torch.cat([self.train_mask[:index], self.train_mask[index+1:]], dim=0)
        self.val_mask = torch.cat([self.val_mask[:index], self.val_mask[index+1:]], dim=0)
        self.test_mask = torch.cat([self.test_mask[:index], self.test_mask[index+1:]], dim=0)

        del self.paper_ids[index]
        del self.node_to_authors[index]

        # Reindex authors dictionary
        self.node_to_authors = {i: self.node_to_authors[i] for i in sorted(self.node_to_authors)}
        self.node_to_authors = {new_i: self.node_to_authors[old_i] for new_i, old_i in enumerate(self.node_to_authors)}

        self.incidence = self._build_incidence_matrix({self.paper_ids[i]: self.node_to_authors[i] for i in range(len(self.paper_ids))})
        print(f"Removed paper {pid}.")


if __name__ == "__main__":
    dataset = HypergraphDataset()