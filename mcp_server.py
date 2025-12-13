#!/usr/bin/env python3
"""
MCP Server for Document RAG
Exposes rag_chat functionality via Model Context Protocol.
"""

import sys
import json
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from vector_store_chroma import ChromaVectorStore
from embedder import Embedder
from retrieve import Retriever
from ingest import load_directory
from chunker import chunk_document

# MCP Protocol Version
PROTOCOL_VERSION = "2024-11-05"

# Global state
store = None
embedder = None
retriever = None
initialized = False


def handle_initialize(params):
    """Handle MCP initialize request."""
    global initialized
    initialized = True
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {"tools": {}},
        "serverInfo": {
            "name": "rag-document-server",
            "version": "1.0.0"
        }
    }


def handle_tools_list():
    """Return available tools."""
    return {
        "tools": [
            {
                "name": "search_documents",
                "description": "Search indexed documents using semantic similarity and optional keyword matching.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        },
                        "top_k": {
                            "type": "number",
                            "description": "Number of results to return (default: 3)",
                            "default": 3
                        },
                        "use_hybrid": {
                            "type": "boolean",
                            "description": "Use hybrid search (vector + BM25)",
                            "default": True
                        },
                        "use_rerank": {
                            "type": "boolean",
                            "description": "Use Cross-Encoder re-ranking for better precision",
                            "default": True
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "index_folder",
                "description": "Index a folder or specific file (PDF, TXT, MD, etc.).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "folder_path": {
                            "type": "string",
                            "description": "Path to folder or file to index"
                        },
                        "priority": {
                            "type": "number",
                            "description": "Priority tier (0=highest, 1=hot, 2=warm)",
                            "default": 0
                        }
                    },
                    "required": ["folder_path"]
                }
            },
            {
                "name": "get_index_status",
                "description": "Get current indexing status and statistics.",
                "inputSchema": {"type": "object", "properties": {}}
            }
        ]
    }


def handle_search_documents(args):
    """Search documents via RAG."""
    global store, retriever, embedder
    
    # Lazy initialize
    if store is None:
        store = ChromaVectorStore()
        embedder = Embedder()
        retriever = Retriever(embedder, store)
    
    query = args["query"]
    top_k = args.get("top_k", 3)
    use_hybrid = args.get("use_hybrid", True)
    use_rerank = args.get("use_rerank", True) # Default to True for better quality
    
    # Retrieve results
    results = retriever.retrieve(query, top_k=top_k, use_hybrid=use_hybrid, use_rerank=use_rerank)
    
    # Format for MCP
    formatted_results = []
    for r in results:
        formatted_results.append({
            "text": r["text"],
            "source": r["metadata"].get("source", "unknown"),
            "page": r["metadata"].get("page", "N/A"),
            "score": round(r.get("score", 0), 3)
        })
    
    return {
        "content": [{
            "type": "text",
            "text": json.dumps({
                "results": formatted_results,
                "query": query,
                "count": len(formatted_results)
            }, indent=2)
        }]
    }


def handle_index_folder(args):
    """Index a folder of documents."""
    global store, embedder
    
    try:
        # Initialize if needed
        if store is None:
            sys.stderr.write("📦 Initializing ChromaVectorStore...\n")
            sys.stderr.flush()
            store = ChromaVectorStore()
            sys.stderr.write("✅ ChromaVectorStore initialized\n")
            sys.stderr.flush()
        
        if embedder is None:
            sys.stderr.write("🧠 Initializing Embedder...\n")
            sys.stderr.flush()
            embedder = Embedder()
            sys.stderr.write(f"✅ Embedder initialized: {embedder}\n")
            sys.stderr.flush()
        
        folder_path = args["folder_path"]
        priority = args.get("priority", 0)
        
        sys.stderr.write(f"📂 Loading documents from {folder_path}...\n")
        sys.stderr.flush()
        
        # Load documents
        docs = load_directory(folder_path)
        sys.stderr.write(f"✅ Loaded {len(docs)} documents\n")
        sys.stderr.flush()
        
        # Chunk all documents
        all_chunks = []
        for doc in docs:
            chunks = chunk_document(doc)
            # Add priority tier to metadata
            for chunk in chunks:
                chunk["metadata"]["tier"] = priority
            all_chunks.extend(chunks)
        
        sys.stderr.write(f"✅ Created {len(all_chunks)} chunks\n")
        sys.stderr.flush()
        
        if len(all_chunks) == 0:
            return {
                "content": [{
                    "type": "text",
                    "text": f"No documents found in {folder_path}"
                }]
            }
        
        # Embed chunks
        texts = [c["text"] for c in all_chunks]
        sys.stderr.write(f"🔮 Embedding {len(texts)} chunks...\n")
        sys.stderr.flush()
        
        embeddings = embedder.embed(texts)
        sys.stderr.write(f"✅ Embeddings created: shape {embeddings.shape}\n")
        sys.stderr.flush()
        
        # Store
        store.add(all_chunks, embeddings)
        sys.stderr.write("✅ Chunks stored in ChromaDB\n")
        sys.stderr.flush()
        
        return {
            "content": [{
                "type": "text",
                "text": f"Indexed {len(all_chunks)} chunks from {len(docs)} documents in {folder_path}"
            }]
        }
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        sys.stderr.write(f"❌ Error in index_folder: {error_details}\n")
        sys.stderr.flush()
        raise Exception(f"Indexing failed: {str(e)}")


def handle_get_index_status(args):
    """Get indexing statistics."""
    global store
    
    if store is None:
        store = ChromaVectorStore()
    
    stats = store.get_stats()
    
    return {
        "content": [{
            "type": "text",
            "text": json.dumps(stats, indent=2)
        }]
    }


async def handle_tools_call(params):
    """Handle tool execution."""
    if not initialized:
        raise Exception("Server not initialized")
    
    name = params["name"]
    args = params.get("arguments", {})
    
    if name == "search_documents":
        return handle_search_documents(args)
    elif name == "index_folder":
        return handle_index_folder(args)
    elif name == "get_index_status":
        return handle_get_index_status(args)
    else:
        raise Exception(f"Unknown tool: {name}")


async def handle_message(msg):
    """Handle incoming MCP message."""
    try:
        message = json.loads(msg)
        
        # Handle requests
        if message.get("id") is not None:
            result = None
            
            if message["method"] == "initialize":
                result = handle_initialize(message.get("params"))
            elif message["method"] == "tools/list":
                result = handle_tools_list()
            elif message["method"] == "tools/call":
                result = await handle_tools_call(message["params"])
            else:
                raise Exception(f"Unknown method: {message['method']}")
            
            send_response(message["id"], result)
        
        # Handle notifications
        elif message.get("method") == "notifications/initialized":
            pass  # Acknowledge
    
    except Exception as error:
        error_code = getattr(error, "code", -32603)
        send_error(message.get("id"), error_code, str(error))


def send_response(msg_id, result):
    """Send JSON-RPC response."""
    response = {
        "jsonrpc": "2.0",
        "id": msg_id,
        "result": result
    }
    print(json.dumps(response), flush=True)


def send_error(msg_id, code, message):
    """Send JSON-RPC error."""
    response = {
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {"code": code, "message": message}
    }
    print(json.dumps(response), flush=True, file=sys.stderr)


# MARK: - Main STDIO Loop

if __name__ == "__main__":
    import asyncio
    
    async def main():
        sys.stderr.write("🔍 RAG Document MCP Server started\n")
        sys.stderr.flush()
        
        buffer = ""
        for line in sys.stdin:
            buffer += line
            
            if '\n' in buffer:
                lines = buffer.split('\n')
                buffer = lines[-1]
                
                for msg in lines[:-1]:
                    if msg.strip():
                        await handle_message(msg)
    
    asyncio.run(main())
