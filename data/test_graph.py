"""
test_arxiv_graph.py

Script to test the arXivHyperGraph and arXivSubGraph classes:
- Loads arXiv data
- Constructs the full hypergraph
- Builds a subgraph
- Adds real, fake, and plausible papers
- Prints statistics for inspection
"""

from paperNodes_graph import arXivHyperGraph
import torch

# Initialize full hypergraph
print("Loading arXivHyperGraph...")
hg = arXivHyperGraph(data_path="arxiv-data/subset_cs_200.json.gz")

# Print initial stats
print("\n--- Hypergraph Summary ---")
print(f"Total papers: {len(hg.paper_ids)}")
print(f"Embedding shape: {hg.x.shape}")
print(f"Number of classes: {len(hg.label_map)}")
print(f"Average authors per paper: {hg.author_mean:.2f}")

# Create a subgraph with 20% dropout
subgraph = hg.construct_subgraph(dropout=0.2)

print("\n--- Initial Subgraph ---")
print(f"Papers in subgraph: {len(subgraph.labels)}")
print(f"Incidence matrix shape: {subgraph.incidence.shape}")
print(f"Unique classes in subgraph: {torch.unique(subgraph.labels).tolist()}")

# Add 100 papers with 30% fake, 30% plausible
print("\nAdding 100 papers: 30% fake, 30% plausible...")
subgraph.add_papers(n=100, fake=0.3, plausible=0.3)

# Print updated stats
print("\n--- Subgraph After Augmentation ---")
print(f"Total papers: {len(subgraph.labels)}")
print(f"Fake papers: {len(subgraph.fake_papers)}")
print(f"Incidence matrix shape: {subgraph.incidence.shape}")
print(f"Label distribution:")
for lbl in torch.unique(subgraph.labels, return_counts=True):
    print(f"  {lbl[0].item()}: {lbl[1].item()} papers")

# Remove fake papers
print("\nRemoving fake papers...")
subgraph.remove_fake_papers()

# Print final stats
print("\n--- Final Subgraph ---")
print(f"Total papers after removal: {len(subgraph.labels)}")
print(f"Incidence matrix shape: {subgraph.incidence.shape}")
print(f"Unique labels: {torch.unique(subgraph.labels).tolist()}")
