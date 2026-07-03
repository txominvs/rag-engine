from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import chromadb
import time, os

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")

while True:
    try:
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=OLLAMA_HOST
        )
        break
    except:
        time.sleep(2)

while True:
    try:
        chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=8000)
        vectorstore = Chroma(
            client=chroma_client,
            collection_name="rag_context_langchain",
            embedding_function=embeddings,
        )
        break
    except:
        time.sleep(2)

# from clients import chroma_collection
# print(chroma_collection.peek())