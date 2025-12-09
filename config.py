# config.py
"""
Global configuration for the RAG project.
Toggle DEBUG to enable/disable debug prints and checks.
"""

OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "mxbai-embed-large"
CHAT_MODEL = "gemma3:4b"

# Debugging toggles
DEBUG = True

# For visualize step (only call when you want)
VISUALIZE_AFTER_BUILD = False

# Chat / LLM model
# Choose one of these based on what you downloaded:
# "gemma3:4b"
# "mistral:7b-instruct"
# "llama3.2:3b-instruct-fp16"
# "llama3.1:8b"
