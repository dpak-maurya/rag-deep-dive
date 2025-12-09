# visualize.py
"""
Visualization tool for RAG embeddings.
Usage:
  python visualize.py index.pkl --umap
  python visualize.py index.pkl --heatmap
  python visualize.py index.pkl --dendrogram
  python visualize.py index.pkl --chunkhist
  python visualize.py index.pkl --retrieval "your query"
"""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from vector_store import VectorStore

###############################
# Optional imports
###############################
try:
    import umap
except ImportError:
    umap = None

try:
    from scipy.cluster.hierarchy import dendrogram, linkage
except ImportError:
    dendrogram = None


#######################################
# Load index
#######################################
def load_index(path):
    store = VectorStore()
    store.load(path)
    chunks, vecs = store.all()
    return chunks, vecs, store


#######################################
# 1. UMAP visualization
#######################################
def plot_umap(vecs):
    if umap is None:
        raise ImportError("UMAP is not installed. pip install umap-learn")

    print("[VIS] Running UMAP dimensionality reduction...")
    proj = umap.UMAP(n_neighbors=15, min_dist=0.1).fit_transform(vecs)

    plt.figure(figsize=(10, 7))
    plt.scatter(proj[:, 0], proj[:, 1], s=8, alpha=0.7)
    plt.title("UMAP Projection of Embeddings")
    plt.show()


#######################################
# 2. Similarity heatmap (50×50 window)
#######################################
def plot_heatmap(vecs):
    print("[VIS] Building similarity heatmap...")
    sim = np.dot(vecs, vecs.T)
    sim /= (
        np.linalg.norm(vecs, axis=1)[:, None]
        * np.linalg.norm(vecs, axis=1)[None, :]
        + 1e-8
    )

    window = sim[:50, :50]

    plt.figure(figsize=(8, 6))
    plt.imshow(window, cmap="viridis")
    plt.colorbar()
    plt.title("Chunk Similarity Heatmap (first 50×50)")
    plt.show()


#######################################
# 3. Dendrogram
#######################################
def plot_dendrogram(vecs):
    if dendrogram is None:
        raise ImportError("SciPy is not installed. pip install scipy")

    print("[VIS] Creating hierarchical clustering dendrogram...")
    Z = linkage(vecs, method="ward")

    plt.figure(figsize=(12, 6))
    dendrogram(Z, truncate_mode="level", p=6)
    plt.title("Hierarchical Clustering of Chunks")
    plt.show()


#######################################
# 4. Chunk length histogram
#######################################
def plot_chunk_hist(chunks):
    lengths = [len(c) for c in chunks]

    plt.hist(lengths, bins=30)
    plt.title("Chunk Length Distribution")
    plt.xlabel("Characters")
    plt.ylabel("Frequency")
    plt.show()


#######################################
# 5. Retrieval visualization (UMAP)
#######################################
def plot_retrieval(store, embedder, query, top_k=5):
    # embed query
    q_vec = embedder.embed([query])[0]

    # compute sims
    vecs = store.vectors
    sims = np.dot(vecs, q_vec) / (
        np.linalg.norm(vecs, axis=1) * np.linalg.norm(q_vec) + 1e-8
    )
    top_i = np.argsort(sims)[::-1][:top_k]

    if umap is None:
        raise ImportError("UMAP is not installed. pip install umap-learn")

    proj = umap.UMAP(n_neighbors=15, min_dist=0.1).fit_transform(vecs)

    plt.figure(figsize=(10, 7))
    plt.scatter(proj[:, 0], proj[:, 1], s=8, alpha=0.3, label="All Chunks")

    pts = proj[top_i]
    plt.scatter(pts[:, 0], pts[:, 1], s=60, color="red", label="Top-K Retrieved")

    plt.title(f"Retrieval Visualization for Query: {query}")
    plt.legend()
    plt.show()


#######################################
# CLI
#######################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index_file", help="Path to index.pkl")
    parser.add_argument("--umap", action="store_true", help="Run UMAP projection")
    parser.add_argument("--heatmap", action="store_true", help="Show similarity heatmap")
    parser.add_argument("--dendrogram", action="store_true", help="Show clustering dendrogram")
    parser.add_argument("--chunkhist", action="store_true", help="Show chunk length histogram")
    parser.add_argument("--retrieval", type=str, help="Visualize retrieval relevance")
    parser.add_argument("--topk", type=int, default=5)

    args = parser.parse_args()

    # Load index
    chunks, vecs, store = load_index(args.index_file)

    # Run chosen visualisation
    if args.umap:
        plot_umap(vecs)

    if args.heatmap:
        plot_heatmap(vecs)

    if args.dendrogram:
        plot_dendrogram(vecs)

    if args.chunkhist:
        plot_chunk_hist(chunks)

    if args.retrieval:
        # Lazy import embedder so visualize.py stays independent
        from embedder import Embedder
        embedder = Embedder()
        plot_retrieval(store, embedder, args.retrieval, top_k=args.topk)


if __name__ == "__main__":
    main()