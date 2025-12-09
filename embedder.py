# embedder.py
import numpy as np
import requests
from config import OLLAMA_BASE_URL, EMBED_MODEL


class Embedder:
    def __init__(self, model: str = EMBED_MODEL):
        self.model = model

    def embed(self, texts):
        """
        Sends a list of texts to the Ollama embedding model.
        Returns: numpy array of shape (N, D)
        """
        url = f"{OLLAMA_BASE_URL}/api/embeddings"

        embeddings = []

        for t in texts:
            payload = {"model": self.model, "prompt": t}
            r = requests.post(url, json=payload)
            if r.status_code != 200:
                raise Exception(f"Embedding failed: {r.text}")

            data = r.json()

            # IMPORTANT: Ollama returns `embedding`, not `embeddings`
            vec = data.get("embedding")
            if vec is None:
                raise KeyError(f"No embedding found in response: {data}")

            embeddings.append(vec)

        return np.array(embeddings, dtype=float)