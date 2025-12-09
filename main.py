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
import time
from tqdm import tqdm
import numpy as np
logging.basicConfig(level=logging.INFO)

from chunker import chunk_text
from embedder import Embedder
from vector_store import VectorStore
from retrieve import Retriever
from config import CHAT_MODEL, OLLAMA_BASE_URL, DEBUG


# ============================================================
# OLLAMA CHAT COMPLETION
# ============================================================
def ask_ollama(model: str, prompt: str) -> str:
    """
    Calls Ollama's chat API.
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
        hits = retriever.retrieve(q, top_k=5, debug=DEBUG)
        rag_context = "\n\n".join([h["chunk"] for h in hits])

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
        if DEBUG:
            print("\n[DEBUG] Full prompt sent to model:\n", prompt[:1000], "...\n")
        answer = ask_ollama(model, prompt)
        print("\nAssistant > " + answer + "\n")

        # Save to memory
        memory.append({"user": q, "assistant": answer})


# ============================================================
# BUILD INDEX
# ============================================================
def build_index(input_file: str, index_file: str):
    print("[DEBUG] Starting index build...")
    print("[DEBUG] Reading input file:", input_file)

    embedder = Embedder()
    store = VectorStore()

    # -------------------------------
    # Read text
    # -------------------------------
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    # -------------------------------
    # Chunk text
    # -------------------------------
    print("\n[DEBUG] Starting chunking...")
    chunks = chunk_text(text)
    print(f"[DEBUG] Total input text length: {len(text)}")
    print(f"[DEBUG] Total chunks created: {len(chunks)}")

    lens = [len(c) for c in chunks]
    print(f"[DEBUG] Chunk lengths | min={min(lens)}, max={max(lens)}, avg={int(sum(lens)/len(lens))}")
    print(f"[DEBUG] Chunking Finished")

    # -------------------------------
    # Request embeddings in batches
    # -------------------------------
    BATCH_SIZE = 16
    all_vecs = []

    print(f"[DEBUG] Requesting embeddings from Ollama in batches of {BATCH_SIZE}...\n")

    start_time = time.time()
    batch_times = []

    # tqdm progress bar
    pbar = tqdm(
        range(0, len(chunks), BATCH_SIZE),
        desc="Embedding Progress"
    )

    for i in pbar:
        batch_start = time.time()

        # Slice batch
        batch = chunks[i:i + BATCH_SIZE]

        # Embed
        batch_vecs = embedder.embed(batch)
        all_vecs.append(batch_vecs)

        # Time measurement
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)

        # Spinner-like animation via tqdm dynamic postfix
        pbar.set_postfix({
            "batch": f"{i//BATCH_SIZE + 1}",
            "size": len(batch),
            "sec": f"{batch_time:.2f}s",
            "avg": f"{np.mean(batch_times):.2f}s"
        })

        # Optional readable line
        preview = batch[0].replace("\n", " ")[:60]
        print(f"\n  [INFO] Batch {i//BATCH_SIZE + 1} done in {batch_time:.2f}s | preview: '{preview}...'")

    # -------------------------------
    # Stack results
    # -------------------------------
    vecs = np.vstack(all_vecs)

    total_time = time.time() - start_time
    print(f"\n[DEBUG] All embeddings complete | shape={vecs.shape}")
    print(f"[DEBUG] Total embedding time: {total_time:.2f} seconds")
    print(f"[DEBUG] Avg time per batch: {np.mean(batch_times):.2f} seconds")
    print(f"[DEBUG] Fastest batch: {np.min(batch_times):.2f}s | Slowest batch: {np.max(batch_times):.2f}s\n")

    # -------------------------------
    # Embedding stats
    # -------------------------------
    first_vec = vecs[0]
    print(
        "[DEBUG] First embedding stats | "
        f"Norm={np.linalg.norm(first_vec):.2f}, "
        f"Mean={first_vec.mean():.3f}, "
        f"Min={first_vec.min():.3f}, "
        f"Max={first_vec.max():.3f}"
    )

    # -------------------------------
    # Store chunks + embeddings
    # -------------------------------
    store.add(chunks, vecs)

    # -------------------------------
    # Save to pickle
    # -------------------------------
    store.save(index_file)

    print(f"[DEBUG] Index build complete:")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Embedding dims: {vecs.shape[1]}")
    print(f"  Saved to: {index_file}")


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
        store = VectorStore()
        store.load(args.path1)
        embedder = Embedder()

        chat_loop(store, embedder, args.model)


if __name__ == "__main__":
    main()