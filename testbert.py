"""
testbert.py
Huub Al

Script to compare SciBERT embeddings with random tensors.
"""
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

def get_scibert_embedding(text):
    """Get SciBERT embedding for a text."""
    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
    
    # Move model to CPU for simplicity
    model = model.to('cpu')
    model.eval()
    
    # Get embedding
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze()  # Get [CLS] token embedding
    
    return embedding

def generate_random_embedding(dim=768):
    """Generate random tensor with same dimensionality as SciBERT."""
    return torch.randn(dim)

def visualize_embeddings(bert_embedding, random_embedding):
    """Visualize embeddings using t-SNE."""
    # Convert to numpy arrays
    bert_np = bert_embedding.numpy().reshape(1, -1)
    random_np = random_embedding.numpy().reshape(1, -1)
    
    # Combine embeddings
    combined = np.vstack([bert_np, random_np])
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(combined)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced[0, 0], reduced[0, 1], c='blue', label='SciBERT', s=100)
    plt.scatter(reduced[1, 0], reduced[1, 1], c='red', label='Random', s=100)
    plt.title('t-SNE visualization of embeddings')
    plt.legend()
    plt.savefig('embedding_comparison.png')
    plt.close()

def analyze_differences(bert_embedding, random_embedding):
    """Analyze and print differences between embeddings."""
    # Convert to numpy for easier analysis
    bert_np = bert_embedding.numpy()
    random_np = random_embedding.numpy()
    
    # Calculate statistics
    cosine_sim = 1 - cosine(bert_np, random_np)  # Convert distance to similarity
    l2_dist = np.linalg.norm(bert_np - random_np)
    mean_diff = np.mean(np.abs(bert_np - random_np))
    std_diff = np.std(np.abs(bert_np - random_np))
    
    print("\nEmbedding Analysis:")
    print(f"Cosine Similarity: {cosine_sim:.4f}")
    print(f"L2 Distance: {l2_dist:.4f}")
    print(f"Mean Absolute Difference: {mean_diff:.4f}")
    print(f"Std of Absolute Differences: {std_diff:.4f}")
    
    # Plot distribution of differences
    plt.figure(figsize=(10, 6))
    plt.hist(np.abs(bert_np - random_np), bins=50)
    plt.title('Distribution of Absolute Differences')
    plt.xlabel('Absolute Difference')
    plt.ylabel('Frequency')
    plt.savefig('difference_distribution.png')
    plt.close()

def main():
    # Sample text
    text = "The transformer architecture has revolutionized natural language processing through its self-attention mechanism."
    
    print("Getting SciBERT embedding...")
    bert_embedding = get_scibert_embedding(text)
    
    print("Generating random embedding...")
    random_embedding = generate_random_embedding()
    
    print("Visualizing embeddings...")
    visualize_embeddings(bert_embedding, random_embedding)
    
    print("Analyzing differences...")
    analyze_differences(bert_embedding, random_embedding)
    
    print("\nDone! Check embedding_comparison.png and difference_distribution.png for visualizations.")

if __name__ == "__main__":
    main() 