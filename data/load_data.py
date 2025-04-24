import os
import gzip
import json


def load_metadata(infile):
    """
    Load metadata saved by all_of_arxiv, as a list of lines of gzip compressed
    json.

    Parameters
    ----------
        infile : str or None
            name of file saved by gzip. If None, one is attempted to be found
            in the expected location with the expected name.

    Returns
    -------
        article_attributes : list
            list of dicts, each of which contains the metadata attributes of
            the ArXiv articles
    """
    fname = infile
    with gzip.open(fname, 'rt', encoding='utf-8') as fin:
        return [json.loads(line) for line in fin.readlines()]

if __name__ == "__main__":
    infile = "/Users/huubal/scriptie/data/arxiv-data/arxiv-metadata-small.json.gz"
    print(load_metadata(infile)[0])