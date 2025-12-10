# ingest.py
import os

ALLOWED_EXT = {".txt", ".md", ".py", ".json"}

def load_directory(path: str):
    """
    Walk a directory and return list of:
    {
        "text": full file text,
        "source": file path,
        "lines": list of individual lines
    }
    """
    docs = []

    for root, dirs, files in os.walk(path):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        for f in files:
            if f.startswith("."):
                continue

            ext = os.path.splitext(f)[1].lower()
            if ext not in ALLOWED_EXT:
                continue

            fp = os.path.join(root, f)

            with open(fp, "r", encoding="utf-8", errors="ignore") as file:
                text = file.read()
                lines = text.split("\n")

            docs.append({
                "text": text,
                "lines": lines,
                "source": fp
            })

    return docs