# chunker.py
"""
Simple text chunker for RAG.
Splits document into manageable sections with paragraph awareness.
"""

from typing import List, Dict

def chunk_document(doc: dict, max_chars=800) -> List[Dict]:
    """
    Splits text into chunks with a maximum size.
    Prefers to break on paragraph boundaries when possible.
    Handles long paragraphs safely.
    Takes a document with:
       { text, source, lines }
    Returns chunks with metadata:
       { text, metadata }
    """

    text = doc["text"]
    source = doc["source"]
    lines = doc["lines"]

    paragraphs = text.split("\n\n")
    chunks = []
    current = ""
    start_line = 0

    line_count = 0

    for p in paragraphs:
        striped_p = p.strip()
        if not striped_p:
            continue

        p_lines = striped_p.count("\n") + 1

        # Hard split if paragraph > max_chars
        if len(striped_p) > max_chars:
            if current.strip():
                chunks.append({
                    "text": current.strip(),
                    "metadata": {
                        "source": source,
                        "start_line": start_line,
                        "end_line": line_count
                    }
                })
                current = ""

            for i in range(0, len(striped_p), max_chars):
                sub = striped_p[i:i+max_chars]
                chunks.append({
                    "text": sub,
                    "metadata": {
                        "source": source,
                        "start_line": line_count,
                        "end_line": line_count + p_lines
                    }
                })

            line_count += p_lines
            continue

        # Otherwise, try adding paragraph to current chunk
        if len(current) + len(striped_p) <= max_chars:
            if current == "":
                start_line = line_count
            current += striped_p + "\n\n"
            line_count += p_lines
        else:
            # Save current chunk
            chunks.append({
                "text": current.strip(),
                "metadata": {
                    "source": source,
                    "start_line": start_line,
                    "end_line": line_count
                }
            })
            current = striped_p + "\n\n"
            start_line = line_count
            line_count += p_lines

    # Final chunk
    if current.strip():
        chunks.append({
            "text": current.strip(),
            "metadata": {
                "source": source,
                "start_line": start_line,
                "end_line": line_count
            }
        })

    return chunks