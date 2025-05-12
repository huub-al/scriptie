# create_hypergraph_dataset.py

import gzip
import json
import torch
import numpy as np
import torch.backends
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
from tqdm import tqdm
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class HypergraphDataset:
    x: torch.Tensor
    y: torch.Tensor
    incidence: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    paper_ids: list
    node_to_authors: dict

def load_papers(file_path):
    papers = []
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            papers.append(json.loads(line))
    return papers

def get_scibert_embeddings(texts, tokenizer, model, device):
    embeddings = []
    for text in tqdm(texts, desc="Embedding abstracts"):
        inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()
        embeddings.append(cls_embedding)
    return torch.stack(embeddings)

def build_incidence_matrix(paper_to_authors):
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

def main():
    file_path = "arxiv-data/subset_cs_2000.json.gz"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    papers = load_papers(file_path)

    paper_ids = []
    abstracts = []
    labels = []
    paper_to_authors = {}

    for paper in papers:
        if not paper.get("abstract"):
            continue
        paper_ids.append(paper["id"])
        abstracts.append(paper["abstract"])
        labels.append(paper["categories"][0])
        paper_to_authors[paper["id"]] = [a.strip() for a in paper["authors"].split(" and ")]

    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased").to(device)
    x = get_scibert_embeddings(abstracts, tokenizer, model, device)

    label_set = sorted(set(labels))
    label_map = {lbl: i for i, lbl in enumerate(label_set)}
    y = torch.tensor([label_map[lbl] for lbl in labels], dtype=torch.long)

    incidence = build_incidence_matrix(paper_to_authors)

    # Split: 60% train, 20% val, 20% test
    indices = np.arange(len(y))
    train_idx, test_val_idx = train_test_split(indices, test_size=0.4, stratify=y, random_state=42)
    val_idx, test_idx = train_test_split(test_val_idx, test_size=0.5, stratify=y[test_val_idx], random_state=42)

    train_mask = torch.zeros(len(y), dtype=torch.bool)
    val_mask = torch.zeros(len(y), dtype=torch.bool)
    test_mask = torch.zeros(len(y), dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    dataset = HypergraphDataset(
        x=x,
        y=y,
        incidence=incidence,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        paper_ids=paper_ids,
        node_to_authors={i: paper_to_authors[pid] for i, pid in enumerate(paper_ids)}
    )

    torch.save(dataset, "arxiv-data/hypergraph_dataset.pt")
    print("Saved hypergraph dataset to hypergraph_dataset.pt")

if __name__ == "__main__":
    main()
