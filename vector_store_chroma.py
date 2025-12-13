# vector_store_chroma.py
"""
ChromaDB-based vector store with hybrid search (Vector + BM25).
Replaces the simple NumPy implementation with production-grade storage.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import numpy as np
from pathlib import Path


class ChromaVectorStore:
    def __init__(self, persist_directory: str = "~/Library/Application Support/Aura/chroma_db"):
        """
        Initialize ChromaDB with persistent storage.
        
        Args:
            persist_directory: Path to store embeddings (default: Aura app support)
        """
        # Expand ~ to full path
        persist_path = str(Path(persist_directory).expanduser())
        Path(persist_path).mkdir(parents=True, exist_ok=True)
        
        print(f"[DEBUG] Initializing ChromaDB at: {persist_path}")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_path,
            settings=Settings(
                anonymized_telemetry=False,  # Privacy-first
                allow_reset=True
            )
        )
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={
                "hnsw:space": "cosine",  # Cosine similarity
                "hnsw:construction_ef": 200,  # Build quality
                "hnsw:search_ef": 100  # Search quality
            }
        )
        
        print(f"[DEBUG] Collection loaded. Current count: {self.collection.count()}")
    
    def add(self, chunk_objects: List[Dict], vectors: np.ndarray):
        """
        Add chunks and embeddings to the store.
        
        Args:
            chunk_objects: List of {text, metadata} dicts
            vectors: NumPy array of embeddings (N, D)
        """
        print(f"\n[DEBUG] Adding {len(chunk_objects)} chunks to ChromaDB...")
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        embeddings = []
        metadatas = []
        
        for i, chunk in enumerate(chunk_objects):
            # Generate unique ID
            source = chunk["metadata"].get("source", "unknown")
            chunk_id = f"{source}_chunk_{i}_{hash(chunk['text'])}"
            
            ids.append(chunk_id)
            documents.append(chunk["text"])
            embeddings.append(vectors[i].tolist())
            
            # Add metadata with tier info
            metadata = chunk["metadata"].copy()
            metadata["tier"] = metadata.get("tier", 1)  # Default to tier 1 (hot)
            metadatas.append(metadata)
        
        # Add to collection (ChromaDB handles deduplication)
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        print(f"[DEBUG] Added {len(ids)} chunks. Total in store: {self.collection.count()}")
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 3,
        tier_filter: Optional[int] = None,
        debug: bool = False
    ) -> List[Dict]:
        """
        Vector similarity search with optional tier filtering.
        
        Args:
            query_vector: Query embedding
            top_k: Number of results to return
            tier_filter: If set, only search this tier (0=priority, 1=hot, 2=warm)
            debug: Print debug info
        
        Returns:
            List of {text, metadata, score} dicts
        """
        # Build metadata filter
        where = None
        if tier_filter is not None:
            where = {"tier": tier_filter}
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        if debug:
            print("\n========== DEBUG: SEARCH ==========")
            print(f"Query vector shape: {query_vector.shape}")
            print(f"Tier filter: {tier_filter}")
            print(f"Results found: {len(results['ids'][0])}")
            print("===================================\n")
        
        # Format results
        formatted = []
        for i in range(len(results["ids"][0])):
            formatted.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1.0 - results["distances"][0][i]  # Convert distance to similarity
            })
        
        return formatted
    
    def hybrid_search(
        self,
        query_text: str,
        query_vector: np.ndarray,
        top_k: int = 10,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3
    ) -> List[Dict]:
        """
        Hybrid search combining vector similarity and BM25 keyword matching.
        
        Args:
            query_text: Original query text (for BM25)
            query_vector: Query embedding (for vector search)
            top_k: Number of candidates to retrieve (before re-ranking)
            vector_weight: Weight for vector similarity (default 0.7)
            bm25_weight: Weight for BM25 score (default 0.3)
        
        Returns:
            Top results after hybrid re-ranking
        """
        # Get top-2K candidates via vector search (cast wide net)
        vector_results = self.search(query_vector, top_k=top_k * 2, debug=False)
        
        # Compute BM25 scores for those candidates
        # (Simplified: keyword overlap, full BM25 would require term frequencies)
        query_terms = set(query_text.lower().split())
        
        for result in vector_results:
            doc_terms = set(result["text"].lower().split())
            
            # Simple keyword overlap score (0-1)
            overlap = len(query_terms & doc_terms)
            bm25_score = overlap / max(len(query_terms), 1)
            
            # Hybrid score
            result["hybrid_score"] = (
                vector_weight * result["score"] +
                bm25_weight * bm25_score
            )
        
        # Re-rank by hybrid score
        vector_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        return vector_results[:top_k]
    
    def get_stats(self) -> Dict:
        """Get collection statistics."""
        return {
            "total_chunks": self.collection.count(),
            "persist_path": self.client.get_settings().persist_directory
        }
    
    def clear(self):
        """Clear all data (for testing or reset)."""
        print("[DEBUG] Clearing ChromaDB collection...")
        self.client.delete_collection("documents")
        self.collection = self.client.get_or_create_collection("documents")
        print("[DEBUG] Collection cleared.")


# Backward compatibility: maintain same interface as original
def VectorStore(*args, **kwargs):
    """Factory function for backward compatibility."""
    return ChromaVectorStore(*args, **kwargs)
