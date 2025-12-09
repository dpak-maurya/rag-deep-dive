# embedder.py
import numpy as np
import requests
from config import OLLAMA_BASE_URL, EMBED_MODEL


class Embedder:
    def __init__(self, model: str = EMBED_MODEL):
        self.model = model
        print(f"[DEBUG] Embedder initialized with model: {model}")

    def embed(self, chunks):
        """
        Sends a list of chunks to the Ollama embedding model.
        Returns: numpy array of shape (N, D)
        """

        

        url = f"{OLLAMA_BASE_URL}/api/embeddings"

        embeddings = []
        

        for chunk in chunks:
            payload = {"model": self.model, "prompt": chunk}
            r = requests.post(url, json=payload)
            if r.status_code != 200:
                raise Exception(f"Embedding failed: {r.text}")

            data = r.json()

            # IMPORTANT: Ollama returns `embedding`, not `embeddings`
            vec = data.get("embedding")
            if vec is None:
                raise KeyError(f"No embedding found in response: {data}")       

            embeddings.append(vec)

        vecs = np.array(embeddings, dtype=float)  

        # Print first vector
        vec = vecs[0]

        return vecs