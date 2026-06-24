import ollama
import chromadb

ollama_client = ollama.Client(host="http://localhost:11434")

chroma_client = chromadb.HttpClient(host="localhost", port=8000)
chroma_collection = chroma_client.get_or_create_collection("rag_context")