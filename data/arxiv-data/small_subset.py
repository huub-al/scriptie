import gzip
from itertools import islice

# Input and output file paths
input_file = 'arxiv-metadata-large.json.gz'
output_file = 'arxiv-metadata-medium.json.gz'

# Write first 50000 entries to output file
print("Writing first 50000 entries...")
with gzip.open(input_file, 'rt', encoding='utf-8') as in_f, \
     gzip.open(output_file, 'wt', encoding='utf-8') as out_f:
    # Read and write first 50000 lines
    for line in islice(in_f, 50000):
        out_f.write(line)

print(f"Successfully created {output_file} with first 50000 entries")
