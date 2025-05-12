import gzip
import json
from collections import defaultdict

# Parameters
input_file = "arxiv-metadata-large.json.gz"
total_samples = 2000
output_file = f"subset_cs_{total_samples}.json.gz"

# Choose the CS subcategories you're interested in
included_subcategories = {"cs.AI", "cs.LG", "cs.CV", "cs.CL", "cs.NE", "cs.CR"}
target_per_subcategory = total_samples // len(included_subcategories)

# Data structures
subcategory_groups = defaultdict(list)
seen_counts = defaultdict(int)

# Read and collect
with gzip.open(input_file, 'rt', encoding='utf-8') as f:
    for count, line in enumerate(f):
        if count % 1000 == 0:
            print(f"Processed lines: {count}")
        paper = json.loads(line)

        if "categories" not in paper:
            continue

        # Look for the first matching cs subcategory
        categories = paper["categories"]
        matching_subcat = next((cat for cat in categories if cat in included_subcategories), None)

        if matching_subcat is None:
            continue

        if seen_counts[matching_subcat] < target_per_subcategory:
            subcategory_groups[matching_subcat].append(paper)
            seen_counts[matching_subcat] += 1

        if all(seen_counts[sub] >= target_per_subcategory for sub in included_subcategories):
            print("Collected sufficient data for all selected cs subcategories.")
            break

# Combine and save
sampled_papers = []
for sub in included_subcategories:
    sampled_papers.extend(subcategory_groups[sub])

with gzip.open(output_file, 'wt', encoding='utf-8') as out_f:
    for paper in sampled_papers:
        out_f.write(json.dumps(paper) + "\n")

print(f"Saved {len(sampled_papers)} CS papers to '{output_file}'.")
