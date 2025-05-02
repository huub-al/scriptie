import torch
from torch_geometric.data import Data
from hypergraph import ArxivHyperGraph
from AllSet.src.models import SetGNN
import argparse
import numpy as np

def load_model(args):
    # Load hypergraph
    print("Loading hypergraph...")
    graph = ArxivHyperGraph(args.data_path)
 
    # Create data object
    data = create_data_object(graph)
    
    # Create and load model
    model = SetGNN(args)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(args.device)
    model.eval()
    
    return model, graph, data

def predict_edge(model, graph, data, paper_id, author):
    """
    Predict whether an edge between a paper and author is plausible
    
    Args:
        model: Trained AllSet model
        graph: ArxivHyperGraph instance
        data: PyTorch Geometric Data object
        paper_id: ID of the paper
        author: Name of the author
        
    Returns:
        float: Probability of the edge being plausible
    """
    # Get paper and author indices
    if paper_id not in graph.paper_to_idx or author not in graph.author_to_idx:
        return 0.0
    
    paper_idx = graph.paper_to_idx[paper_id]
    author_idx = graph.author_to_idx[author]
    
    # Get model predictions
    with torch.no_grad():
        out = model(data)
        
    # Get the probability for this edge
    edge_prob = torch.sigmoid(out[paper_idx, author_idx]).item()
    
    return edge_prob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/arxiv-data/arxiv-metadata-small.json.gz',
                      help='Path to the arXiv data file')
    parser.add_argument('--model_path', type=str, default='models/allset_model.pt',
                      help='Path to the trained model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use')
    
    args = parser.parse_args()
    
    # Load model and data
    model, graph, data = load_model(args)
    
    # Example prediction
    paper_id = list(graph.edge_dict.keys())[0]  # Get first paper
    author = list(graph.author_dict.keys())[0]  # Get first author
    
    prob = predict_edge(model, graph, data, paper_id, author)
    print(f"Probability of edge between paper {paper_id} and author {author}: {prob:.4f}")

if __name__ == '__main__':
    main() 