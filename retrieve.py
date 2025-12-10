# retrieve.py
import logging
import numpy as np


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

    # ===============================================================
    # PUBLIC METHOD
    # ===============================================================
    def retrieve(self, query, top_k=5, debug=False):
        """
        Normal mode:
            - delegates to vector store search()

        Debug mode:
            - performs full transparent manual retrieval
            - prints stats + histograms
            - shows full chunk + metadata in results
        """
        q_vec = self.embedder.embed([query])[0]

        # ----------------------------------------------------------
        # NORMAL FAST MODE
        # ----------------------------------------------------------
        if not debug:
            return self.store.search(q_vec, top_k=top_k, debug=False)

        # ----------------------------------------------------------
        # DEBUG MODE — FULL INSPECTION
        # ----------------------------------------------------------
        print("\n" + "=" * 60)
        print("🔍 RAG RETRIEVAL DEBUG MODE")
        print("=" * 60)
        print(f"[QUERY] {query}")

        chunks, vectors = self.store.all()  # chunks = {"text":..., "metadata":...}
        N = len(chunks)

        # -------------------------
        # Query vector stats
        # -------------------------
        print("\n[DEBUG] Query Embedding Stats:")
        print(f"  Shape: {q_vec.shape}")
        print(f"  Norm: {np.linalg.norm(q_vec):.4f}")
        print(f"  Mean: {q_vec.mean():.5f}")
        print(f"  Min/Max: {q_vec.min():.4f} / {q_vec.max():.4f}")
        print(f"  First 10 dims: {q_vec[:10]}")

        # -------------------------
        # Cosine similarities
        # -------------------------
        dot = np.dot(vectors, q_vec)
        denom = np.linalg.norm(vectors, axis=1) * np.linalg.norm(q_vec) + 1e-8
        sims = dot / denom

        print("\n[DEBUG] Cosine Similarity Distribution:")
        print(f"  Min:  {sims.min():.4f}")
        print(f"  Max:  {sims.max():.4f}")
        print(f"  Mean: {sims.mean():.4f}")
        print(f"  Std:  {sims.std():.4f}")

        # -------------------------
        # Plot histogram
        # -------------------------
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 3))
        plt.hist(sims, bins=20)
        plt.title("Cosine Similarity Distribution")
        plt.xlabel("similarity")
        plt.ylabel("count")
        plt.tight_layout()
        plt.show()

        # -------------------------
        # Histogram buckets
        # -------------------------
        hist, edges = np.histogram(sims, bins=10)
        print("\n  Histogram (10 bins):")
        for i in range(len(hist)):
            print(f"     {edges[i]:.2f}–{edges[i+1]:.2f}: {hist[i]}")

        # -------------------------
        # TOP K
        # -------------------------
        top_i = np.argsort(sims)[::-1][:top_k]
        results = []

        print(f"\n[DEBUG] Top {top_k} matches:")
        for rank, idx in enumerate(top_i, 1):
            score = float(sims[idx])
            chunk_obj = chunks[idx]
            text = chunk_obj["text"]
            metadata = chunk_obj["metadata"]

            preview = text.replace("\n", " ")[:150]

            print(f"\n  #{rank} | score={score:.4f} | index={idx}")
            print(f"      Preview: {preview}...")
            print(f"      Metadata: {metadata}")

            results.append({
                "text": text,
                "metadata": metadata,
                "score": score
            })

        print("=" * 60 + "\n")

        # -------------------------
        # TOP-K BAR CHART
        # -------------------------
        top_scores = [float(sims[i]) for i in top_i]
        labels = [f"Chunk {i}" for i in top_i]

        plt.figure(figsize=(8, 4))
        plt.barh(labels, top_scores)
        plt.xlabel("Cosine Similarity")
        plt.title("Top-K Retrieved Chunk Scores")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

        # Logging
        logging.info(
            "RAG Debug retrieve | query='%s' | results=%s",
            query,
            [{"score": r["score"], "metadata": r["metadata"]} for r in results]
        )

        return results