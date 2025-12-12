# chunker.py
"""
Smarter chunker for RAG with:
 - sliding window + overlap for generic text
 - Python AST-based splitting (functions/classes) for code
 - returns list of chunk objects:
    { "text": "...", "metadata": { "source": "...", "chunk_index": i, "start_line": a, "end_line": b } }
"""

from typing import List, Dict, Optional, Tuple
import math
import re

# For Python AST splitting
import ast

# Helpers ---------------------------------------------------------------------
def _line_offsets(text: str) -> List[int]:
    """Return the char offsets where each line starts (0-based)."""
    offsets = [0]
    for m in re.finditer(r"\n", text):
        offsets.append(m.end())
    return offsets

def _chars_to_lines(offsets: List[int], char_start: int, char_end: int) -> Tuple[int, int]:
    """
    Convert char offsets to approximate (start_line, end_line) 0-based indices.
    Returns 1-based line numbers to be human-friendly.
    """
    # find last offset <= char_start
    start_line = 1
    end_line = 1
    for i, off in enumerate(offsets):
        if off <= char_start:
            start_line = i + 1
        if off <= char_end:
            end_line = i + 1
    return start_line, end_line

# Text chunker (sliding window) ----------------------------------------------
def sliding_window_chunks(text: str, source: str, chunk_size: int = 800, overlap: int = 200) -> List[Dict]:
    """
    Sliding window chunking with overlap.
    chunk_size = max characters per chunk
    overlap = number of characters to overlap between adjacent chunks
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")

    offsets = _line_offsets(text)
    chunks = []
    n = len(text)
    step = chunk_size - overlap
    idx = 0
    chunk_index = 0

    while idx < n:
        end = min(idx + chunk_size, n)
        chunk_text = text[idx:end].strip()
        if chunk_text:
            start_line, end_line = _chars_to_lines(offsets, idx, end)
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": source,
                    "chunk_index": chunk_index,
                    "start_line": start_line,
                    "end_line": end_line
                }
            })
            chunk_index += 1
        idx += step

    return chunks

# Paragraph-aware chunker with overlap ---------------------------------------
def paragraph_chunks(text: str, source: str, max_chars: int = 800, overlap_chars: int = 200) -> List[Dict]:
    """
    Try to split on paragraph boundaries first; then apply sliding-window on long paragraphs.
    Keeps overlap between resulting chunks (~overlap_chars).
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    full = "\n\n".join(paragraphs)
    # Use sliding window on the joined paragraphs to ensure consistent overlap
    return sliding_window_chunks(full, source, chunk_size=max_chars, overlap=overlap_chars)

# Python AST-based chunker ---------------------------------------------------
def python_ast_chunks(text: str, source: str, max_chars:int = 1200, overlap_chars:int = 200) -> List[Dict]:
    """
    Parse Python source and chunk by AST nodes (Module -> top-level classes and functions).
    Each AST node chunk attempts to include the full node; for very large nodes, we fall back to sliding.
    Returns list of {text, metadata}.
    """
    try:
        tree = ast.parse(text)
    except Exception:
        # fallback to paragraph_chunks if parse fails
        return paragraph_chunks(text, source, max_chars=max_chars, overlap_chars=overlap_chars)

    offsets = _line_offsets(text)
    chunks = []
    chunk_index = 0

    # gather ranges from top-level FunctionDef and ClassDef nodes (use end_lineno if available)
    node_ranges: List[Tuple[int,int,ast.AST]] = []
    for node in tree.body:
        if hasattr(node, "lineno"):
            start = getattr(node, "lineno", None)
            # end line: Python 3.8+ has end_lineno, else approximate by node.body last lineno
            end = getattr(node, "end_lineno", None)
            if end is None:
                # try to guess end line from node children
                if hasattr(node, "body") and isinstance(node.body, list) and node.body:
                    last = node.body[-1]
                    end = getattr(last, "lineno", start)
                else:
                    end = start
            node_ranges.append((start, end, node))

    # If no top-level defs, fallback to paragraph_chunks
    if not node_ranges:
        return paragraph_chunks(text, source, max_chars=max_chars, overlap_chars=overlap_chars)

    # Extract source for each node (convert lines to char offsets)
    lines = text.splitlines(keepends=True)
    total_lines = len(lines)
    for start, end, node in node_ranges:
        # clamp
        s = max(1, start)
        e = min(total_lines, end)
        # convert to char offsets (0-based)
        char_start = sum(len(lines[i]) for i in range(0, s-1))
        char_end = sum(len(lines[i]) for i in range(0, e))
        node_text = text[char_start:char_end].strip()
        if not node_text:
            continue

        # if node_text is small enough, accept as a chunk
        if len(node_text) <= max_chars:
            chunks.append({
                "text": node_text,
                "metadata": {
                    "source": source,
                    "chunk_index": chunk_index,
                    "start_line": s,
                    "end_line": e
                }
            })
            chunk_index += 1
        else:
            # node too large — split via sliding window inside this node
            inner_chunks = sliding_window_chunks(node_text, source, chunk_size=max_chars, overlap=overlap_chars)
            # adjust start/end lines for inner chunks: approximate by offsetting from s
            # compute char offsets for node_text to convert idx -> lines
            node_offsets = _line_offsets(node_text)
            for ic in inner_chunks:
                # compute approximate start/end in local node_text
                # convert text slices to lines using node_offsets + s
                # for simplicity compute lines via counting newlines in prefix
                local_start_char = 0  # sliding_window_chunks stores metadata using global offsets, but here we recalc
                # convert first occurrence of ic["text"] inside node_text (best-effort)
                find_idx = node_text.find(ic["text"][:50])
                if find_idx != -1:
                    # estimate line numbers
                    start_line_off, end_line_off = _chars_to_lines(node_offsets, find_idx, find_idx + len(ic["text"]))
                    abs_start = s - 1 + (start_line_off)
                    abs_end = s - 1 + (end_line_off)
                else:
                    abs_start = s
                    abs_end = e

                chunks.append({
                    "text": ic["text"],
                    "metadata": {
                        "source": source,
                        "chunk_index": chunk_index,
                        "start_line": int(abs_start),
                        "end_line": int(abs_end)
                    }
                })
                chunk_index += 1

    # After node-based chunks, we may also want to include file-level preamble or trailing text not in nodes
    # Add any top-level text segments between nodes (best-effort)
    return chunks

# Master function ------------------------------------------------------------
def chunk_document(doc: dict,
                   max_chars: int = 800,
                   overlap: int = 200,
                   prefer_ast_for_code: bool = True) -> List[Dict]:
    """
    doc: {"text": str, "source": path, "lines": [...]}
    Returns list of chunk objects with metadata.
    """
    text = doc.get("text", "")
    source = doc.get("source", "<unknown>")

    # language hint by extension
    ext = None
    if "." in source:
        ext = source.rsplit(".", 1)[1].lower()

    # If Python and prefer_ast_for_code, try AST splitting
    if prefer_ast_for_code and ext == "py":
        return python_ast_chunks(text, source, max_chars=max_chars, overlap_chars=overlap)

    # Otherwise do paragraph-aware sliding window
    return paragraph_chunks(text, source, max_chars=max_chars, overlap_chars=overlap)