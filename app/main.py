import ollama

client = ollama.Client(host="http://localhost:11434")

llm_response = client.chat(
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

embedding_response = client.embeddings(
    model="nomic-embed-text",
    prompt="Some sample text"
).embedding

print(embedding_response)