# vector_store.py
"""
Simple in-memory vector store for RAG.
Stores chunks + numpy embeddings.
"""

import numpy as np
from typing import List
import pickle


class VectorStore:
    def __init__(self):
        self.chunks = []
        self.vectors = None  # numpy array shape (N, D)
        print("[DEBUG] VectorStore initialized.")

    def add(self, chunk_objects, vectors):
        """
        chunk_objects = list of { text, metadata }
        vectors = numpy array
        """
        print("\n[DEBUG] Storing chunks + vectors...")
        print(f"[DEBUG] Chunks: {len(chunk_objects)}")
        print(f"[DEBUG] Vectors shape: {vectors.shape}")

        self.chunks.extend(chunk_objects)

        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])

    def all(self):
        """
        Returns all chunks + embeddings.
        """
        return self.chunks, self.vectors
    
        # -----------------------------------------------------
    # SAVE INDEX
    # -----------------------------------------------------
    def save(self, path="index/index.pkl"):
        print(f"[DEBUG] Saving vector store → {path}")
        with open(path, "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "vectors": self.vectors
            }, f)
        print("[DEBUG] Save complete.")

    # -----------------------------------------------------
    # LOAD INDEX
    # -----------------------------------------------------
    def load(self, path="index/index.pkl"):
        print(f"[DEBUG] Loading vector store ← {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.chunks = data["chunks"]
            self.vectors = data["vectors"]
        print("[DEBUG] Load complete. Chunks:", len(self.chunks))

    # -----------------------------------------------------
    # SEARCH
    # -----------------------------------------------------
    def search(self, query_vector, top_k=3, debug=False):
        """
        Performs cosine similarity search.
        Returns list of dicts: { chunk, score }
        """
        if self.vectors is None or len(self.chunks) == 0:
            raise ValueError("Vector store is empty. Build index first.")

        query_vec = np.array(query_vector)

        # Normalize
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        store_norm = self.vectors / (
            np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-10
        )

        # Cosine similarity
        scores = np.dot(store_norm, query_norm)

        # Get top K
        top_idx = np.argsort(scores)[::-1][:top_k]

        if debug:
            print("\n========== DEBUG: SEARCH ==========")
            print("Query Vector (first 5 dims):", query_vec[:5])
            print("Similarity Scores:", scores)
            print("Top K indices:", top_idx)
            print("====================================\n")

        results = []
        for idx in top_idx:
            score = float(scores[idx])
            chunk_obj = self.chunks[idx]
            text = chunk_obj["text"]
            metadata = chunk_obj["metadata"]
            results.append({
                "text": text,
                "metadata": metadata,
                "score": score
            })

        return results