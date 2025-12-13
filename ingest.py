# ingest.py
import os

ALLOWED_EXT = {".txt", ".md", ".py", ".json", ".pdf"}

def load_path(path: str):
    """
    Load a single file or walk a directory and return list of:
    {
        "text": full file text,
        "source": file path,
        "lines": list of individual lines
    }
    """
    docs = []

    if os.path.isfile(path):
        # Handle single file
        files_to_process = [path]
        base_dir = os.path.dirname(path)
    else:
        # Handle directory
        files_to_process = []
        for root, dirs, files in os.walk(path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for f in files:
                if not f.startswith("."):
                    files_to_process.append(os.path.join(root, f))

    for fp in files_to_process:
        ext = os.path.splitext(fp)[1].lower()
        if ext not in ALLOWED_EXT:
            continue
        
        # Handle PDF files
        if ext == ".pdf":
            try:
                import PyPDF2
                with open(fp, "rb") as file:
                    pdf = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() + "\n"
                    lines = text.split("\n")
            except Exception as e:
                print(f"⚠️ Failed to read PDF {fp}: {e}")
                continue
        else:
            # Handle text files
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as file:
                    text = file.read()
                    lines = text.split("\n")
            except Exception as e:
                print(f"⚠️ Failed to read {fp}: {e}")
                continue

        # Prepend filename to text content so it is embedded and searchable
        filename = os.path.basename(fp)
        final_text = f"File: {filename}\n\n{text}"

        docs.append({
            "text": final_text,
            "lines": lines,
            "source": fp
        })

    return docs

# Alias for backward compatibility
load_directory = load_path