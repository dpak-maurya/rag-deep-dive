
import sys
import os
from retrieve import Retriever

# Mock classes to avoid full ChromaDB/Embedder dependency for logic test
class MockEmbedder:
    def embed(self, texts):
        import numpy as np
        # Return random vectors
        return np.random.rand(len(texts), 384)

class MockStore:
    def search(self, query_vec, top_k=5, debug=False):
        # Return dummy candidates with low initial scores
        return [
            {"text": "Apple is a fruit.", "metadata": {"id": 1}, "score": 0.5},
            {"text": "Python is a programming language.", "metadata": {"id": 2}, "score": 0.5},
            {"text": "Sky is blue.", "metadata": {"id": 3}, "score": 0.5},
            {"text": "Apples are delicious red fruits.", "metadata": {"id": 4}, "score": 0.5},
            {"text": "Coding in Python is fun.", "metadata": {"id": 5}, "score": 0.5}
        ]
    
    def hybrid_search(self, query_text, query_vector, top_k=5):
        return self.search(query_vector, top_k)

def test_reranking():
    print("🧪 Testing Cross-Encoder Re-ranking...")
    
    embedder = MockEmbedder()
    store = MockStore()
    retriever = Retriever(embedder, store)
    
    if retriever.cross_encoder is None:
        print("❌ Cross-Encoder failed to load (check internet/dependencies).")
        return

    query = "fruit"
    print(f"\n🔍 Query: '{query}'")
    
    results = retriever.retrieve(query, top_k=3, use_rerank=True, debug=True)
    
    print("\n✅ Results:")
    for res in results:
        print(f"  - Score: {res['score']:.4f} | Text: {res['text']}")
        
    # Validation
    top_text = results[0]["text"]
    if "Apple" in top_text or "fruit" in top_text:
        print("\n✅ SUCCESS: Relevant text bubbled to the top!")
    else:
        print("\n❌ FAILURE: Re-ranking didn't prioritize relevant text.")

if __name__ == "__main__":
    test_reranking()
