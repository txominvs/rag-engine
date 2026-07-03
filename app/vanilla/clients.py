import ollama
import chromadb
import time, os

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")

while True:
    try:
        ollama_client = ollama.Client(host=OLLAMA_HOST)
        break
    except:
        time.sleep(2)

while True:
    try:
        chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=8000)
        chroma_collection = chroma_client.get_or_create_collection("rag_context")
        break
    except:
        time.sleep(2)