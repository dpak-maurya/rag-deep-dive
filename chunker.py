# chunker.py
"""
Simple text chunker for RAG.
Splits document into manageable sections with paragraph awareness.
"""

from typing import List

def chunk_text(text: str, max_chars: int = 800) -> List[str]:
    """
    Splits text into chunks with a maximum size.
    Prefers to break on paragraph boundaries when possible.
    Handles long paragraphs safely.
    """
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for p in paragraphs:
        p = p.strip()
        if not p:
            continue

        # If paragraph alone is too large → force-split it
        if len(p) > max_chars:
            # finalize current chunk first
            if current.strip():
                chunks.append(current.strip())
                current = ""

            # split long paragraph into hard chunks
            for i in range(0, len(p), max_chars):
                sub = p[i:i + max_chars]
                chunks.append(sub.strip())

            continue

        # normal case: add paragraph if it fits
        if len(current) + len(p) <= max_chars:
            current += p + "\n\n"
        else:
            # close current chunk
            if current.strip():
                chunks.append(current.strip())

            current = p + "\n\n"

    # final chunk
    if current.strip():
        print(current.strip())
        chunks.append(current.strip())

    return chunks