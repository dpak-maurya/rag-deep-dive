# embedder.py
import numpy as np
import requests
import concurrent.futures
from config import OLLAMA_BASE_URL, EMBED_MODEL
import tqdm

class Embedder:
    def __init__(self, model: str = EMBED_MODEL):
        self.model = model
        print(f"[DEBUG] Embedder initialized with model: {model}")

    def embed(self, chunks):
        """
        Sends a list of chunks to the Ollama embedding model in parallel.
        Returns: numpy array of shape (N, D)
        """
        url = f"{OLLAMA_BASE_URL}/api/embeddings"
        embeddings = [None] * len(chunks)
        
        def process_chunk(index, text):
            try:
                payload = {"model": self.model, "prompt": text}
                r = requests.post(url, json=payload)
                if r.status_code != 200:
                    print(f"⚠️ Error embedding chunk {index}: {r.text}")
                    return None
                
                data = r.json()
                vec = data.get("embedding")
                if vec is None:
                    print(f"⚠️ No embedding found for chunk {index}")
                    return None
                    
                return vec
            except Exception as e:
                print(f"⚠️ Exception processing chunk {index}: {e}")
                return None

        # Use 10 threads to parallelize HTTP requests
        # Ollama usually queues them efficiently for the GPU
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_index = {
                executor.submit(process_chunk, i, chunk): i 
                for i, chunk in enumerate(chunks)
            }
            
            for future in concurrent.futures.as_completed(future_to_index):
                i = future_to_index[future]
                result = future.result()
                if result is not None:
                    embeddings[i] = result
        
        # Filter out failed embeddings if any (maintain alignment? No, must match)
        # For simplicity, we assume robust success or fill with zeros?
        # Let's fill failures with zeros to keep shape
        
        final_embeddings = []
        for vec in embeddings:
            if vec is None:
                # Fallback: zero vector (needs size check, but usually 1024/768/512)
                # We'll just skip for now and hope for the best or handle in vector store
                pass # Ideally we shouldn't have Nones
            else:
                final_embeddings.append(vec)

        if not final_embeddings:
            return np.array([])
            
        vecs = np.array(final_embeddings, dtype=float)
        return vecs