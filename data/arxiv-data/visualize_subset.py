import gzip
import json
from collections import Counter
import matplotlib.pyplot as plt

# Parameters
input_file = "subset_cs_20000.json.gz"

# Count top-level categories
category_counts = Counter()

with gzip.open(input_file, 'rt', encoding='utf-8') as f:
    for line in f:
        paper = json.loads(line)
        if "categories" in paper:
            top_category = paper["categories"][0]
            category_counts[top_category] += 1

# Plot
categories = list(category_counts.keys())
counts = [category_counts[cat] for cat in categories]

plt.figure(figsize=(10, 6))
plt.bar(categories, counts, color='skyblue')
plt.xlabel('Top-level Category')
plt.ylabel('Number of Papers')
plt.title('Distribution of Categories in Sampled ArXiv Subset')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save or show
plt.savefig("category_distribution.png")
plt.show()
