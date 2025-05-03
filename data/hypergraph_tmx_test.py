"""
hypergraph_tmx_test.py
Huub Al

Test script for the hypergraph implementation.
"""
import torch
import numpy as np
from hypergraph_tmx import ArxivHyperGraph

def test_hypergraph(input_file):
    """
    Test the hypergraph implementation.
    
    Args:
        input_file (str): Path to the input JSON.GZ file
    """
    print("Creating main hypergraph...")
    graph = ArxivHyperGraph(input_file)
    
    print("\nGenerating subgraph with 10% dropout...")
    subgraph = graph.gen_subgraph(dropout=0.9)
    
    # Test 1: Check if all authors are present in subgraph
    print("\nTest 1: Checking if all authors are present...")
    all_authors_present = len(subgraph.author_dict) == len(graph.author_dict)
    print(f"All authors present: {all_authors_present}")
    
    # Test 2: Check if subgraph has fewer edges than main graph
    print("\nTest 2: Checking edge count...")
    main_edges = len(graph.edge_dict)
    sub_edges = len(subgraph.edges_left)
    print(f"Main graph edges: {main_edges}")
    print(f"Subgraph edges: {sub_edges}")
    print(f"Dropout rate achieved: {(main_edges - sub_edges) / main_edges:.2%}")
    
    # Test 3: Check if all authors have at least one paper
    print("\nTest 3: Checking if all authors have at least one paper...")
    authors_without_papers = []
    for author, papers in subgraph.author_dict.items():
        if not any(paper in subgraph.edges_left for paper in papers):
            authors_without_papers.append(author)
    print(f"Authors without papers: {len(authors_without_papers)}")
    
    # Test 4: Check node features shape and content
    print("\nTest 4: Checking node features...")
    print(f"Node features shape: {subgraph.node_features.shape}")
    print(f"Number of authors: {len(subgraph.author_dict)}")
    print(f"Any zero embeddings: {torch.any(torch.all(subgraph.node_features == 0, dim=1))}")
    
    # Test 5: Check incidence matrix
    print("\nTest 5: Checking incidence matrix...")
    print(f"Incidence matrix shape: {subgraph.sub_H.shape}")
    print(f"Number of non-zero elements: {subgraph.sub_H._nnz()}")
    
    # Test 6: Generate plausible fake edge
    print("\nTest 6: Testing plausible fake edge generation...")
    paper_embedding, selected_authors = subgraph.generate_plausible_fake_edge()
    print(f"Generated plausible fake paper with {len(selected_authors)} authors")
    print(f"Authors in plausible fake paper:{selected_authors}")
    print(f"Paper embedding shape: {paper_embedding.shape}")

    # Test 7: Generate fake edge
    print("\nTest 7: Testing fake edge generation...")
    paper_embedding, selected_authors = subgraph.generate_fake_edge()
    print(f"Generated fake paper with {len(selected_authors)} authors")
    print(f"Authors in fake paper: {selected_authors}")
    print(f"Paper embedding shape: {paper_embedding.shape}")

    # Test 8: Generate real edges
    print("\nTest 8: Testing real edge generation...")
    real_edges = subgraph.generate_real_edge()
    print(f"Number of real edges (dropped papers with authors): {len(real_edges)}")
    if real_edges:
        # Show stats for first edge as example
        first_edge_embedding, first_edge_authors = real_edges[0]
        print("\nExample edge:")
        print(f"Number of authors: {len(first_edge_authors)}")
        print(f"Author indices: {first_edge_authors}")
        print(f"Embedding shape: {first_edge_embedding.shape}")
        
        # Show distribution of author counts
        author_counts = [len(authors) for _, authors in real_edges]
        print(f"\nAuthor count distribution:")
        print(f"Min authors: {min(author_counts)}")
        print(f"Max authors: {max(author_counts)}")
        print(f"Mean authors: {sum(author_counts)/len(author_counts):.2f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python hypergraph_tmx_test.py <input_file>")
        sys.exit(1)
    
    test_hypergraph(sys.argv[1]) 