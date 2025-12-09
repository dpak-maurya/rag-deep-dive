# retrieve.py
import logging
import numpy as np


class Retriever:
    def __init__(self, embedder, store):
        self.embedder = embedder
        self.store = store

    def retrieve(self, query, top_k=5):
        """
        Embed query -> cosine similarity -> return top k chunks.
        """
        chunks, vectors = self.store.all()

        # embed the query
        q_vec = self.embedder.embed([query])[0]

        # cosine similarity
        sims = np.dot(vectors, q_vec) / (
            np.linalg.norm(vectors, axis=1) * np.linalg.norm(q_vec) + 1e-8
        )

        # top K indices
        top_i = np.argsort(sims)[::-1][:top_k]

        # return text + score
        logging.info("query=%s, top_scores=%s",query,[chunks[i] for i in top_i])
        return [{"text": chunks[i], "score": float(sims[i])} for i in top_i]