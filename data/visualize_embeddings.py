import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from hypergraph import ArxivHyperGraph
import seaborn as sns
from collections import Counter

def get_cs_subcategory(category):
    """
    Get the Computer Science subcategory from a category string.
    
    Args:
        category (str): arXiv category code
        
    Returns:
        str: CS subcategory or None if not a CS paper
    """
    # Split multiple categories if present
    categories = category.split()
    
    # Check each category
    for cat in categories:
        if 'cs.' in cat.lower():
            # Extract the subcategory (e.g., 'cs.AI' -> 'AI')
            return cat.split('.')[-1].upper()
    
    return None

# Load the hypergraph
print("Loading hypergraph...")
graph = ArxivHyperGraph('/Users/huubal/scriptie/data/arxiv-data/arxiv-metadata-small.json.gz')

# Extract embeddings and categories
print("Extracting embeddings and categories...")
embeddings = []
categories = []
for paper_id, paper_data in graph.edge_dict.items():
    embeddings.append(paper_data['embedding'])
    categories.append(paper_data['category'])

# Convert to numpy arrays
embeddings = np.array(embeddings)
categories = np.array(categories)

# Filter for CS papers and get their subcategories
cs_papers = []
cs_subcategories = []
for i, cat in enumerate(categories):
    subcategory = get_cs_subcategory(cat)
    if subcategory is not None:
        cs_papers.append(embeddings[i])
        cs_subcategories.append(subcategory)

if not cs_papers:
    print("No Computer Science papers found!")
    exit()

cs_papers = np.array(cs_papers)
cs_subcategories = np.array(cs_subcategories)

# Get subcategory counts
subcategory_counts = Counter(cs_subcategories)
print("\nCS subcategory distribution:")
for subcategory, count in subcategory_counts.most_common():
    print(f"{subcategory}: {count} papers")

# Reduce dimensionality to 2D using PCA
print("\nPerforming PCA...")
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(cs_papers)

# Create a color map for subcategories
unique_subcategories = list(subcategory_counts.keys())
colors = sns.color_palette('husl', n_colors=len(unique_subcategories))
subcategory_to_color = dict(zip(unique_subcategories, colors))

# Create the plot
plt.figure(figsize=(12, 8))

# Plot each point with its subcategory color
for subcategory in unique_subcategories:
    mask = cs_subcategories == subcategory
    plt.scatter(
        embeddings_2d[mask, 0],
        embeddings_2d[mask, 1],
        c=[subcategory_to_color[subcategory]],
        label=f"{subcategory} ({subcategory_counts[subcategory]})",
        alpha=0.6
    )

plt.title('PCA of Computer Science Paper Embeddings')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Save the plot
plt.savefig('cs_paper_embeddings_pca.png', bbox_inches='tight', dpi=300)
print("\nPlot saved as 'cs_paper_embeddings_pca.png'")

# Print explained variance
print(f"\nExplained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.2f}") 