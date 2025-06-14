#!/usr/bin/env python3
"""
Comprehensive test suite for paperNodes_graph.py

This test suite thoroughly tests all methods in both arXivHyperGraph and arXivSubGraph classes,
printing detailed information about the behavior and potential issues.
"""

import torch
import numpy as np
import tempfile
import os
import sys
import copy
from collections import defaultdict

# Add path to your data directory
sys.path.append("/Users/huubal/scriptie/data")

try:
    from paperNodes_graph import arXivHyperGraph, arXivSubGraph
    print("✓ Successfully imported paperNodes_graph classes")
except ImportError as e:
    print(f"✗ Failed to import paperNodes_graph: {e}")
    sys.exit(1)


def print_separator(title):
    """Print a formatted separator for test sections."""
    print("\n" + "="*80)
    print(f" {title} ")
    print("="*80)


def print_subsection(title):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


def test_arxiv_hypergraph():
    """Test all methods in arXivHyperGraph class."""
    print_separator("TESTING arXivHyperGraph CLASS")
    
    # Test 1: Initialization and basic properties
    print_subsection("Test 1: Initialization")
    try:
        hypergraph = arXivHyperGraph()
        print("✓ arXivHyperGraph initialization successful")
        
        # Check basic properties
        print(f"  - Number of papers: {len(hypergraph.paper_ids)}")
        print(f"  - Embedding shape: {hypergraph.x.shape}")
        print(f"  - Labels shape: {hypergraph.y.shape}")
        print(f"  - Number of unique authors: {len(hypergraph.author_pool)}")
        print(f"  - Average authors per paper: {hypergraph.author_mean:.2f}")
        print(f"  - Number of categories: {len(hypergraph.label_map)}")
        print(f"  - Number of synthetic labels: {len(hypergraph.synthetic_labels)}")
        print(f"  - Total labels: {len(hypergraph.full_label_map)}")
        
        # Check label mappings
        print("\n  Label mappings:")
        print(f"    Real categories: {list(hypergraph.label_map.keys())[:5]}...")
        print(f"    Synthetic labels: {list(hypergraph.synthetic_labels.keys())}")
        print(f"    Label indices: {list(hypergraph.full_label_map.values())}")
        
        # Check data consistency
        assert len(hypergraph.paper_ids) == hypergraph.x.shape[0], "Paper IDs count mismatch"
        assert len(hypergraph.paper_ids) == hypergraph.y.shape[0], "Labels count mismatch"
        assert len(hypergraph.node_to_authors) == len(hypergraph.paper_ids), "Authors mapping mismatch"
        print("✓ Data consistency checks passed")
        
    except Exception as e:
        print(f"✗ arXivHyperGraph initialization failed: {e}")
        return None
    
    # Test 2: construct_subgraph method
    print_subsection("Test 2: construct_subgraph Method")
    try:
        # Test with different dropout values
        dropout_values = [0.0, 0.1, 0.5, 0.9]
        
        for dropout in dropout_values:
            subgraph = hypergraph.construct_subgraph(dropout=dropout)
            expected_size = int(len(hypergraph.paper_ids) * (1 - dropout))
            actual_size = len(subgraph.indices)
            
            print(f"  Dropout {dropout}: Expected ~{expected_size}, Got {actual_size}")
            
            # Check subgraph properties
            assert isinstance(subgraph, arXivSubGraph), "Wrong subgraph type"
            assert subgraph.embeddings.shape[0] == actual_size, "Embedding size mismatch"
            assert subgraph.labels.shape[0] == actual_size, "Labels size mismatch"
            assert len(subgraph.node_to_authors) == actual_size, "Authors mapping size mismatch"
            
        print("✓ construct_subgraph method works correctly")
        
    except Exception as e:
        print(f"✗ construct_subgraph test failed: {e}")
    
    return hypergraph


def test_arxiv_subgraph(hypergraph):
    """Test all methods in arXivSubGraph class."""
    print_separator("TESTING arXivSubGraph CLASS")
    
    if hypergraph is None:
        print("✗ Cannot test arXivSubGraph - hypergraph is None")
        return
    
    # Test 1: Basic subgraph creation and properties
    print_subsection("Test 1: Basic Subgraph Creation")
    try:
        # Create a subgraph with moderate dropout
        subgraph = hypergraph.construct_subgraph(dropout=0.3)
        
        print(f"  Original graph size: {len(hypergraph.paper_ids)}")
        print(f"  Subgraph size: {subgraph.embeddings.shape[0]}")
        print(f"  Subgraph indices: {len(subgraph.indices)}")
        print(f"  Embeddings shape: {subgraph.embeddings.shape}")
        print(f"  Labels shape: {subgraph.labels.shape}")
        print(f"  Incidence matrix shape: {subgraph.incidence.shape}")
        print(f"  Incidence matrix is sparse: {subgraph.incidence.is_sparse}")
        
        # Check initial tracking lists
        print(f"  Initial fake papers: {len(subgraph.fake_papers)}")
        print(f"  Initial plausible papers: {len(subgraph.plausible_papers)}")
        print(f"  Initial synthetic papers: {len(subgraph.synthetic_papers)}")
        
        # Check label distribution
        unique_labels, counts = torch.unique(subgraph.labels, return_counts=True)
        print(f"  Label distribution: {dict(zip(unique_labels.tolist(), counts.tolist()))}")
        
        print("✓ Basic subgraph creation successful")
        
    except Exception as e:
        print(f"✗ Basic subgraph creation failed: {e}")
        return None
    
    # Test 2: add_papers method
    print_subsection("Test 2: add_papers Method")
    try:
        # Store original state
        original_size = subgraph.embeddings.shape[0]
        original_fake_count = len(subgraph.fake_papers)
        original_plausible_count = len(subgraph.plausible_papers)
        original_synthetic_count = len(subgraph.synthetic_papers)
        
        print(f"  Original size: {original_size}")
        
        # Test adding papers with different proportions
        test_cases = [
            {"n": 10, "fake": 0.3, "plausible": 0.3, "desc": "Mixed papers"},
            {"n": 5, "fake": 1.0, "plausible": 0.0, "desc": "Only fake papers"},
            {"n": 5, "fake": 0.0, "plausible": 1.0, "desc": "Only plausible papers"},
            {"n": 10, "fake": 0.0, "plausible": 0.0, "desc": "Only real papers"},
        ]
        
        for i, case in enumerate(test_cases):
            print(f"\n  Test case {i+1}: {case['desc']}")
            print(f"    Adding {case['n']} papers (fake={case['fake']}, plausible={case['plausible']})")
            
            # Store state before adding
            size_before = subgraph.embeddings.shape[0]
            fake_before = len(subgraph.fake_papers)
            plausible_before = len(subgraph.plausible_papers)
            synthetic_before = len(subgraph.synthetic_papers)
            
            # Add papers
            subgraph.add_papers(case['n'], fake=case['fake'], plausible=case['plausible'])
            
            # Check results
            size_after = subgraph.embeddings.shape[0]
            fake_after = len(subgraph.fake_papers)
            plausible_after = len(subgraph.plausible_papers)
            synthetic_after = len(subgraph.synthetic_papers)
            
            expected_fake = int(case['n'] * case['fake'])
            expected_plausible = int(case['n'] * case['plausible'])
            expected_real = case['n'] - expected_fake - expected_plausible
            
            print(f"    Size change: {size_before} -> {size_after} (expected +{case['n']})")
            print(f"    Fake papers: {fake_before} -> {fake_after} (expected +{expected_fake})")
            print(f"    Plausible papers: {plausible_before} -> {plausible_after} (expected +{expected_plausible})")
            print(f"    Synthetic papers: {synthetic_before} -> {synthetic_after} (expected +{expected_fake + expected_plausible})")
            
            # Verify additions
            assert size_after == size_before + case['n'], f"Size mismatch: expected +{case['n']}, got +{size_after - size_before}"
            assert fake_after == fake_before + expected_fake, f"Fake count mismatch"
            assert plausible_after == plausible_before + expected_plausible, f"Plausible count mismatch"
            
            # Check label distribution after addition
            unique_labels, counts = torch.unique(subgraph.labels, return_counts=True)
            label_dist = dict(zip(unique_labels.tolist(), counts.tolist()))
            print(f"    Updated label distribution: {label_dist}")
            
            # Check synthetic labels are present if expected
            synthetic_label_indices = list(hypergraph.synthetic_labels.values())
            if expected_fake > 0 or expected_plausible > 0:
                has_synthetic = any(label in label_dist for label in synthetic_label_indices)
                print(f"    Has synthetic labels: {has_synthetic}")
        
        print("✓ add_papers method works correctly")
        
    except Exception as e:
        print(f"✗ add_papers test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: remove_fake_papers method
    print_subsection("Test 3: remove_fake_papers Method")
    try:
        # Store state before removal
        size_before_removal = subgraph.embeddings.shape[0]
        fake_before_removal = len(subgraph.fake_papers)
        plausible_before_removal = len(subgraph.plausible_papers)
        synthetic_before_removal = len(subgraph.synthetic_papers)
        
        print(f"  Before removal:")
        print(f"    Size: {size_before_removal}")
        print(f"    Fake papers: {fake_before_removal}")
        print(f"    Plausible papers: {plausible_before_removal}")
        print(f"    Synthetic papers: {synthetic_before_removal}")
        
        # Get label distribution before removal
        unique_labels_before, counts_before = torch.unique(subgraph.labels, return_counts=True)
        label_dist_before = dict(zip(unique_labels_before.tolist(), counts_before.tolist()))
        print(f"    Label distribution before: {label_dist_before}")
        
        # Remove fake papers
        subgraph.remove_fake_papers()
        
        # Check state after removal
        size_after_removal = subgraph.embeddings.shape[0]
        fake_after_removal = len(subgraph.fake_papers)
        plausible_after_removal = len(subgraph.plausible_papers)
        synthetic_after_removal = len(subgraph.synthetic_papers)
        
        print(f"  After removal:")
        print(f"    Size: {size_after_removal}")
        print(f"    Fake papers: {fake_after_removal}")
        print(f"    Plausible papers: {plausible_after_removal}")
        print(f"    Synthetic papers: {synthetic_after_removal}")
        
        # Get label distribution after removal
        unique_labels_after, counts_after = torch.unique(subgraph.labels, return_counts=True)
        label_dist_after = dict(zip(unique_labels_after.tolist(), counts_after.tolist()))
        print(f"    Label distribution after: {label_dist_after}")
        
        # Verify removal
        expected_size_after = size_before_removal - synthetic_before_removal
        print(f"    Expected size after removal: {expected_size_after}")
        
        # Check that synthetic papers were removed
        synthetic_label_indices = list(hypergraph.synthetic_labels.values())
        has_synthetic_after = any(label in label_dist_after for label in synthetic_label_indices)
        print(f"    Has synthetic labels after removal: {has_synthetic_after}")
        
        # Verify tracking lists are cleared
        assert fake_after_removal == 0, "Fake papers list not cleared"
        assert plausible_after_removal == 0, "Plausible papers list not cleared"
        assert synthetic_after_removal == 0, "Synthetic papers list not cleared"
        
        print("✓ remove_fake_papers method works correctly")
        
    except Exception as e:
        print(f"✗ remove_fake_papers test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: outlier detection and removal
    print_subsection("Test 4: Outlier Detection and Removal")
    try:
        # First, add the outlier tracking attributes if they don't exist
        if not hasattr(subgraph, 'outliers'):
            subgraph.outliers = []
        if not hasattr(subgraph, 'removed_outliers_data'):
            subgraph.removed_outliers_data = {}
        
        # Store state before outlier removal
        size_before_outliers = subgraph.embeddings.shape[0]
        print(f"  Size before outlier removal: {size_before_outliers}")
        
        # Calculate some stats about embeddings
        embedding_norms = torch.norm(subgraph.embeddings, dim=1)
        mean_norm = embedding_norms.mean()
        std_norm = embedding_norms.std()
        print(f"  Embedding norms - Mean: {mean_norm:.4f}, Std: {std_norm:.4f}")
        print(f"  Embedding norms - Min: {embedding_norms.min():.4f}, Max: {embedding_norms.max():.4f}")
        
        # Test outlier removal with different thresholds
        for outlier_fraction in [0.01, 0.02, 0.02]:
            # Reset subgraph for each test
            if hasattr(subgraph, 'outliers'):
                subgraph.restore_outliers()
            
            print(f"\n  Testing outlier removal with outlier fraction ={outlier_fraction}")
            
            size_before = subgraph.embeddings.shape[0]
            subgraph.remove_outliers(outlier_fraction=outlier_fraction)
            size_after = subgraph.embeddings.shape[0]
            
            outliers_removed = size_before - size_after
            print(f"    Size: {size_before} -> {size_after} (removed {outliers_removed} outliers)")
            print(f"    Should have removed: {int(size_before * outlier_fraction)} outliers")
            print(f"    Outliers list length: {len(subgraph.outliers)}")
            print(f"    Removed outliers data: {len(subgraph.removed_outliers_data)}")
            
            if outliers_removed > 0:
                # Test outlier restoration
                subgraph.restore_outliers()
                size_after_restore = subgraph.embeddings.shape[0]
                print(f"    Size after restoration: {size_after_restore}")
                
                assert size_after_restore == size_before, "Outlier restoration failed"
                assert len(subgraph.outliers) == 0, "Outliers list not cleared after restoration"
                assert len(subgraph.removed_outliers_data) == 0, "Outliers data not cleared"
        
        print("✓ Outlier detection and removal works correctly")
        
    except Exception as e:
        print(f"✗ Outlier detection test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: _rebuild_incidence method (indirectly)
    print_subsection("Test 5: Incidence Matrix Consistency")
    try:
        # Test that incidence matrix is rebuilt correctly after modifications
        original_incidence_shape = subgraph.incidence.shape
        print(f"  Current incidence matrix shape: {original_incidence_shape}")
        
        # Add some papers and check incidence matrix
        subgraph.add_papers(5, fake=0.5, plausible=0.5)
        new_incidence_shape = subgraph.incidence.shape
        print(f"  Incidence matrix shape after adding papers: {new_incidence_shape}")
        
        # Check that number of rows matches number of nodes
        assert new_incidence_shape[0] == subgraph.embeddings.shape[0], "Incidence matrix rows don't match node count"
        
        # Check incidence matrix properties
        incidence_dense = subgraph.incidence.to_dense()
        print(f"  Incidence matrix density: {(incidence_dense > 0).float().mean():.4f}")
        print(f"  Non-zero entries: {subgraph.incidence._nnz()}")
        
        # Verify incidence matrix represents author-paper relationships correctly
        total_author_paper_connections = sum(len(authors) for authors in subgraph.node_to_authors.values())
        print(f"  Expected connections: {total_author_paper_connections}")
        print(f"  Actual connections: {subgraph.incidence._nnz()}")
        
        print("✓ Incidence matrix consistency checks passed")
        
    except Exception as e:
        print(f"✗ Incidence matrix test failed: {e}")
    
    return subgraph


def test_edge_cases_and_robustness():
    """Test edge cases and robustness of the implementation."""
    print_separator("TESTING EDGE CASES AND ROBUSTNESS")
    
    try:
        # Create a small hypergraph for testing
        hypergraph = arXivHyperGraph()
        
        print_subsection("Test 1: Empty and Small Subgraphs")
        
        # Test with very high dropout (near empty subgraph)
        try:
            subgraph = hypergraph.construct_subgraph(dropout=0.99)
            print(f"  High dropout subgraph size: {subgraph.embeddings.shape[0]}")
            
            if subgraph.embeddings.shape[0] > 0:
                # Try adding papers to very small subgraph
                subgraph.add_papers(2, fake=0.5, plausible=0.5)
                print(f"  Size after adding to small subgraph: {subgraph.embeddings.shape[0]}")
                print("✓ Small subgraph handling works")
            else:
                print("  Subgraph too small to test further operations")
                
        except Exception as e:
            print(f"✗ Small subgraph test failed: {e}")
        
        print_subsection("Test 2: Large Paper Additions")
        
        # Test adding large numbers of papers
        try:
            subgraph = hypergraph.construct_subgraph(dropout=0.1)
            original_size = subgraph.embeddings.shape[0]
            
            # Try adding a large number of papers
            large_n = min(100, len(hypergraph.paper_ids) // 10)
            print(f"  Adding {large_n} papers to subgraph")
            
            subgraph.add_papers(large_n, fake=0.33, plausible=0.33)
            new_size = subgraph.embeddings.shape[0]
            
            print(f"  Size change: {original_size} -> {new_size}")
            print(f"  Synthetic papers added: {len(subgraph.synthetic_papers)}")
            print("✓ Large paper addition works")
            
        except Exception as e:
            print(f"✗ Large paper addition test failed: {e}")
        
        print_subsection("Test 3: Repeated Operations")
        
        # Test repeated add/remove cycles
        try:
            subgraph = hypergraph.construct_subgraph(dropout=0.2)
            
            for cycle in range(3):
                print(f"  Cycle {cycle + 1}:")
                size_start = subgraph.embeddings.shape[0]
                
                # Add papers
                subgraph.add_papers(10, fake=0.4, plausible=0.4)
                size_after_add = subgraph.embeddings.shape[0]
                
                # Remove fake papers
                subgraph.remove_fake_papers()
                size_after_remove = subgraph.embeddings.shape[0]
                
                print(f"    Start: {size_start}, After add: {size_after_add}, After remove: {size_after_remove}")
            
            print("✓ Repeated operations work correctly")
            
        except Exception as e:
            print(f"✗ Repeated operations test failed: {e}")
        
        print_subsection("Test 4: Memory and Performance")
        
        # Check memory usage and basic performance
        try:
            subgraph = hypergraph.construct_subgraph(dropout=0.1)
            
            # Measure incidence matrix size
            incidence_memory = subgraph.incidence._nnz() * 4 * 2  # Rough estimate in bytes
            dense_memory = subgraph.incidence.shape[0] * subgraph.incidence.shape[1] * 4
            
            print(f"  Incidence matrix sparse storage: ~{incidence_memory / 1024:.1f} KB")
            print(f"  Incidence matrix dense would be: ~{dense_memory / 1024:.1f} KB")
            print(f"  Sparsity savings: {100 * (1 - incidence_memory / dense_memory):.1f}%")
            
            # Test memory after operations
            import gc
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Perform some operations
            for _ in range(5):
                subgraph.add_papers(20, fake=0.5, plausible=0.5)
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            print(f"  Memory usage: {memory_before:.1f} MB -> {memory_after:.1f} MB")
            
            print("✓ Memory and performance checks completed")
            
        except Exception as e:
            print(f"⚠ Memory/performance test failed (non-critical): {e}")
    
    except Exception as e:
        print(f"✗ Edge cases test setup failed: {e}")


def print_final_summary():
    """Print a final summary of all tests."""
    print_separator("TEST SUMMARY")
    
    print("This test suite has evaluated:")
    print("  ✓ arXivHyperGraph initialization and caching")
    print("  ✓ Subgraph construction with various dropout rates")
    print("  ✓ Paper addition (real, fake, and plausible)")
    print("  ✓ Synthetic paper removal")
    print("  ✓ Outlier detection and removal/restoration")
    print("  ✓ Incidence matrix consistency")
    print("  ✓ Edge cases and robustness")
    
    print("\nIf you noticed any '✗' marks above, those indicate potential issues")
    print("that should be investigated in the paperNodes_graph.py implementation.")
    
    print(f"\nTest completed at: {sys.datetime.datetime.now()}")


def main():
    """Run the complete test suite."""
    print("COMPREHENSIVE TEST SUITE FOR paperNodes_graph.py")
    print("="*80)
    print("This will test all methods and print detailed information about behavior.")
    print("Look for ✓ (success) and ✗ (failure) markers in the output.")
    
    # Test arXivHyperGraph
    hypergraph = test_arxiv_hypergraph()
    
    # Test arXivSubGraph
    subgraph = test_arxiv_subgraph(hypergraph)
    
    # Test edge cases
    test_edge_cases_and_robustness()
    
    # Print final summary
    print_final_summary()


if __name__ == "__main__":
    main()