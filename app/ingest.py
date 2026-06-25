from clients import ollama_client, chroma_collection
import hashlib

def ingest_texts(texts: list[str]) -> None:
    new_ids=[]
    new_embeddings=[]
    new_texts=[]

    for text in texts:
        text = text.strip()
        
        hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        existing = chroma_collection.get(ids=hash)['ids']
        if not len(existing):
            print(f"Adding text: {text[:20].strip()}...")

            embedding = ollama_client.embeddings(
                model="nomic-embed-text",
                prompt=text
            ).embedding

            new_ids.append(hash)
            new_embeddings.append(embedding)
            new_texts.append(text)

    if len(new_ids):
        chroma_collection.add(
            ids=new_ids,
            embeddings=new_embeddings,
            documents=new_texts
        )

def remove_docs() -> None:
    all_ids = chroma_collection.get()['ids']
    if all_ids:
        chroma_collection.delete(ids=all_ids)

if __name__ == "__main__":
    texts = """
    The quick brown fox jumps over the lazy dog.
    Some random text for testing purposes.
    Another line of text to test the embedding and retrieval process.
    This is a sample document for the Chroma collection.
    """.strip().splitlines()

    print("Running example ingestion...")
    ingest_texts(texts)