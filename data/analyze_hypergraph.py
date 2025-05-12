# analyze_hypergraph.py

import torch
import statistics
from collections import Counter
from paperNodes_graph import HypergraphDataset

def load_hypergraph(file_path):
    """
    Loads the HypergraphDataset object from the given file path.

    Args:
        file_path (str): Path to the saved dataset (.pt file).

    Returns:
        HypergraphDataset: Loaded dataset object.
    """
    return torch.load(file_path)

def analyze_authorship(dataset):
    """
    Analyzes authorship statistics in the hypergraph.

    Args:
        dataset (HypergraphDataset): The dataset to analyze.
    """
    author_counts = [len(authors) for authors in dataset.node_to_authors.values()]
    
    most_authors = max(author_counts)
    fewest_authors = min(author_counts)
    mean_authors = statistics.mean(author_counts)
    median_authors = statistics.median(author_counts)
    mode_authors = statistics.mode(author_counts)

    # Find papers with the most authors
    most_author_indices = [i for i, count in enumerate(author_counts) if count == most_authors]
    most_author_papers = [dataset.paper_ids[i] for i in most_author_indices]

    # Count total unique authors
    all_authors = [author for authors in dataset.node_to_authors.values() for author in authors]
    unique_authors = set(all_authors)
    author_frequency = Counter(all_authors)

    print("=== Hypergraph Authorship Statistics ===")
    print(f"Total Papers: {len(dataset.paper_ids)}")
    print(f"Total Unique Authors: {len(unique_authors)}")
    print(f"Most Authors on a Single Paper: {most_authors}")
    print(f"Fewest Authors on a Paper: {fewest_authors}")
    print(f"Mean Authors per Paper: {mean_authors:.2f}")
    print(f"Median Authors per Paper: {median_authors}")
    print(f"Mode Authors per Paper: {mode_authors}")
    print(f"Papers with Most Authors: {most_author_papers}")
    print("\nTop 10 Most Frequent Authors:")
    for author, count in author_frequency.most_common(10):
        print(f"{author}: {count} papers")


if __name__ == "__main__":
    dataset_path = "arxiv-data/hypergraph_dataset.pt"
    dataset = load_hypergraph(dataset_path)
    analyze_authorship(dataset)
