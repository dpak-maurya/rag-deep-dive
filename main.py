# main.py
"""
RAG CLI tool:
- Build index from a document
- Chat using RAG + memory
"""

import argparse
import pickle
import requests
import logging
logging.basicConfig(level=logging.INFO)

from chunker import chunk_text
from embedder import Embedder
from vector_store import VectorStore
from retrieve import Retriever
from config import CHAT_MODEL, OLLAMA_BASE_URL


# ============================================================
# OLLAMA CHAT COMPLETION
# ============================================================
def ask_ollama(model: str, prompt: str) -> str:
    """
    Calls Ollama's chat API (better than /generate).
    Returns the assistant's reply.
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }

    r = requests.post(url, json=payload)
    r.raise_for_status()

    data = r.json()
    return data["message"]["content"]


# ============================================================
# CHAT LOOP
# ============================================================
def chat_loop(store: VectorStore, embedder: Embedder, model: str):
    retriever = Retriever(embedder, store)
    memory = []
    MAX_MEMORY = 8

    print(f"🔵 Chatting with local model: {model}")
    print("Type 'exit' to quit.\n")

    while True:
        q = input("You > ").strip()
        if q.lower() == "exit":
            break

        # -------------------------------
        # RAG RETRIEVAL
        # -------------------------------
        hits = retriever.retrieve(q, top_k=5)
        rag_context = "\n\n".join([h["text"] for h in hits])

        # -------------------------------
        # MEMORY FORMAT
        # -------------------------------
        mem = ""
        for t in memory[-MAX_MEMORY:]:
            mem += f"User: {t['user']}\nAssistant: {t['assistant']}\n\n"

        # -------------------------------
        # FINAL PROMPT
        # -------------------------------
        prompt = f"""
You are a helpful assistant.

Below is the conversation memory, retrieved document context, and the new question.

Use the memory to maintain the conversation flow.
Use the RAG context ONLY for factual grounding.

---

### MEMORY
{mem}

---

### RAG CONTEXT
{rag_context}

---

### QUESTION
{q}

---

### ANSWER
"""

        # -------------------------------
        # LOCAL OLLAMA ANSWER
        # -------------------------------
        answer = ask_ollama(model, prompt)
        print("\nAssistant > " + answer + "\n")

        # Save to memory
        memory.append({"user": q, "assistant": answer})


# ============================================================
# BUILD INDEX
# ============================================================
def build_index(input_file: str, index_file: str):
    print(f"📘 Building index from: {input_file}")

    embedder = Embedder()
    store = VectorStore()

    # Read text
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Chunk
    chunks = chunk_text(text)
    print(f"🔹 Created {len(chunks)} chunks")

    # Embeddings
    vecs = embedder.embed(chunks)
    print(f"🔹 Generated embeddings: {vecs.shape}")

    # Save to store
    store.add(chunks, vecs)

    # Save file
    with open(index_file, "wb") as f:
        pickle.dump(store, f)

    print(f"✅ Index saved: {index_file}")


# ============================================================
# CLI HANDLER
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "chat"])
    parser.add_argument("path1", help="Input document OR index file")
    parser.add_argument("path2", nargs="?", help="Index output file (build only)")
    parser.add_argument("--model", default=CHAT_MODEL)

    args = parser.parse_args()

    if args.command == "build":
        if not args.path2:
            raise ValueError("For 'build', you must provide: input_file index_file")
        build_index(args.path1, args.path2)

    elif args.command == "chat":
        with open(args.path1, "rb") as f:
            store = pickle.load(f)

        embedder = Embedder()
        chat_loop(store, embedder, args.model)


if __name__ == "__main__":
    main()