# 🧠 rag-deep-dive
A step-by-step, from-scratch implementation of a Retrieval-Augmented Generation (RAG) system — built for learning, experimentation, and understanding every internal component deeply.

This project does **not** rely on heavy frameworks like LangChain or LlamaIndex.  
Everything is built manually so you can see how RAG works end-to-end:

Text → Chunker → Embedder → Vector Store → Retriever → LLM → Answer

---

## 🚀 Features (Current State)

### ✔️ **1. Document Chunking**
- Splits large documents into clean, paragraph-aware chunks.
- Max token/char limit per chunk.
- Produces ~800–character chunks.

**File:** `chunker.py`

---

### ✔️ **2. Embedding Generation**
- Uses a local or small embedding model (OpenAI-compatible API or similar).
- Converts each chunk into a **1024-dimensional embedding vector**.
- Handles:
  - batching
  - normalization
  - embedding persistence

**File:** `embedder.py`

---

### ✔️ **3. Simple Vector Store**
- Stores embeddings + chunks in memory.
- Saves and loads them using `pickle`.
- Supports cosine similarity search using NumPy.

**File:** `vector_store.py`

---

### ✔️ **4. Retriever**
- Encodes user queries using the same embedding model.
- Computes similarity with stored embeddings.
- Returns **top-k chunks**.

**File:** `retrieve.py`

---

### ✔️ **5. Chat Loop**
- Accepts user input from CLI.
- Retrieves context.
- Sends context + query to an LLM (OpenAI/Local model).
- Generates an answer.

**File:** `main.py`

---

## 📁 Project Structure

```plaintext
📦 rag-deep-dive  
├── 📄 main.py  
├── 📄 chunker.py  
├── 📄 embedder.py  
├── 📄 vector_store.py  
├── 📄 retrieve.py  
│
├── 📂 data  
│   └── 📄 your_docs.txt  
│
├── 📂 index  
│   └── 🗂️ index.pkl  
│
└── 📄 README.md
```



---

## 🧰 Requirements
    python >= 3.10
    numpy
    openai
    tqdm

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


## ▶️ Running the System

1. Build the index (chunk + embed + store)

    ```bash 
    python main.py --build --file data/your_docs.txt
    ```


2. Run chat mode

    ```bash 
    python main.py --chat
    ```



## 💾 Vector Index

Embeddings and metadata are saved here:

```index/index.pkl```

Contains:
- list of chunk texts
- embeddings matrix (N × 1024)
- metadata

This allows you to load the index instantly without recomputing embeddings.


## 🔍 Current Limitations (before next commit)
- No advanced debugging tools yet.
- No visualization of similarity or vectors.
- No Chroma or FAISS—using a simple in-memory store for learning.
- Pipeline is intentionally simple for step-by-step understanding.



## 🛠️ Next Planned Steps (future commits)

You will add:
    •	Debug logs at each step
    (chunk sizes, embedding shapes, similarity scores)
    •	PCA/2D visualization of embeddings
    •	Inspect each retrieved chunk
    •	Optionally swap in:
    •	FAISS
    •	ChromaDB
    •	better embedding models

These will be added as new commits and branches.



## 📜 License

MIT License — free to use and modify.


## ⭐ Motivation

This project is designed to help developers truly understand how RAG works under the hood:
	•	How embeddings are generated
	•	How similarity search operates
	•	How chunking affects retrieval
	•	How LLMs combine retrieved context with queries

Instead of treating RAG as a black box, this repo reveals every piece step-by-step.