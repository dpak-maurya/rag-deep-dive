# vector_store.py
"""
Simple in-memory vector store for RAG.
Stores chunks + numpy embeddings.
"""

import numpy as np
from typing import List


class VectorStore:
    def __init__(self):
        self.chunks = []
        self.vectors = None  # numpy array shape (N, D)

    def add(self, chunks: List[str], vectors: np.ndarray):
        """
        Adds text chunks + embedding matrix.
        """
        self.chunks.extend(chunks)

        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])

    def all(self):
        """
        Returns all chunks + embeddings.
        """
        return self.chunks, self.vectors