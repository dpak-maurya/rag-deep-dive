# retrieve.py
import logging
import numpy as np
from sentence_transformers import CrossEncoder

class Retriever:
    """
    Retriever that supports chunk objects of the form:
        {
            "text": "...chunk text...",
            "metadata": {...}
        }

    This class works with:
    - embedder.embed([text])  → np.ndarray
    - store.search(query_vec, top_k, debug)
    - store.all() → (chunks, vectors)
    """

    def __init__(self, embedder, store):
        self.embedder = embedder
        self.store = store
        try:
            # Initialize Cross-Encoder for re-ranking
            # "ms-marco-MiniLM-L-6-v2" is fast and effective for passage ranking
            print("[DEBUG] Loading Refiner model (Cross-Encoder)...")
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            print("[DEBUG] Refiner model loaded.")
        except Exception as e:
            print(f"⚠️ Failed to load Cross-Encoder: {e}")
            self.cross_encoder = None

    # ===============================================================
    # PUBLIC METHOD
    # ===============================================================
    def retrieve(self, query, top_k=5, use_hybrid=True, use_rerank=True, debug=False):
        """
        Retrieves documents using vector/hybrid search and optionally re-ranks them.
        
        Args:
            query (str): The search query.
            top_k (int): Number of final results to return.
            use_hybrid (bool): Whether to use hybrid search (Vector + BM25).
            use_rerank (bool): Whether to use Cross-Encoder re-ranking.
            debug (bool): Print debug stats.
        """
        
        # 1. Fetch Candidates
        # If re-ranking, we fetch MORE candidates first (e.g. 5x top_k)
        initial_k = top_k * 5 if use_rerank and self.cross_encoder else top_k
        
        q_vec = self.embedder.embed([query])[0]
        candidates = []

        if use_hybrid and hasattr(self.store, 'hybrid_search'):
            candidates = self.store.hybrid_search(
                query_text=query,
                query_vector=q_vec,
                top_k=initial_k
            )
        else:
            candidates = self.store.search(q_vec, top_k=initial_k, debug=debug)

        # 2. Re-rank Candidates (if enabled)
        if use_rerank and self.cross_encoder:
            if not candidates:
                return []
                
            print(f"[DEBUG] Re-ranking {len(candidates)} candidates for: '{query}'")
            
            # Prepare pairs for Cross-Encoder: [[query, doc1], [query, doc2], ...]
            # We use a limited window of text for speed (first 1000 chars)
            pairs = [[query, doc["text"][:1000]] for doc in candidates]
            
            # Predict scores
            scores = self.cross_encoder.predict(pairs)
            
            # update scores in candidates
            for i, score in enumerate(scores):
                candidates[i]["score"] = float(score) # Cross-Encoder score replaces vector score
                candidates[i]["metadata"]["rerank_score"] = float(score)
            
            # Sort by new score (descending)
            candidates.sort(key=lambda x: x["score"], reverse=True)
            
            # Slice top_k
            results = candidates[:top_k]
            
            if debug:
                print("\n[DEBUG] Top Re-ranked Results:")
                for i, res in enumerate(results):
                    print(f"  #{i+1}: {res['score']:.4f} | {res['metadata'].get('source', 'unknown')}")
        else:
            results = candidates

        # ----------------------------------------------------------
        # DEBUG OUTPUT (Optional)
        # ----------------------------------------------------------
        if debug and not use_rerank:
             print("\n" + "=" * 60)
             print("🔍 RAG RETRIEVAL DEBUG MODE")
             print("=" * 60)
             print(f"[QUERY] {query}")
        
        # Logging
        logging.info(
            "RAG retrieve | query='%s' | rerank=%s | results=%d",
            query,
            use_rerank,
            len(results)
        )

        return results