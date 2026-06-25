from clients import ollama_client, chroma_collection

# Sample LLM query
llm_response = ollama_client.chat(
    model="qwen2.5:0.5b",
    messages=
    [
        {
            "role": "user",
            "content": "Say hello in a single sentence."
        }
    ]
).message.content

print(llm_response)

# Sample embedding generation
original_text = "Some sample text"

embedding_response = ollama_client.embeddings(
    model="nomic-embed-text",
    prompt=original_text
).embedding

print(embedding_response)

# Sample insertion into DB
chroma_collection.add(
    ids="0",
    embeddings=embedding_response,
    documents=original_text
)

# Sample retrival from DB
documents = chroma_collection.query(
    query_embeddings=[embedding_response],
    n_results=1
)["documents"][0]

print(documents)