from clients import ollama_client, chroma_collection

def query_rag(question: str) -> str:

    print(f"{question = }")

    question_embedding = ollama_client.embeddings(
        model="nomic-embed-text",
        prompt=question
    ).embedding

    documents = chroma_collection.query(
        query_embeddings=[question_embedding],
        n_results=1
    )["documents"][0]

    context = ' '.join(documents)

    print(f"{context = }")

    llm_response = ollama_client.chat(
        model="phi3:3.8b",
        messages=
        [
            {
                "role": "user",
                "content": f"""Answer the question using ONLY the information from the context below.
If the answer is not present in the context, simply say "I don't know".

Context: {' '.join(documents)}
Question: {question}
Answer:"""
            }
        ]
    ).message.content

    return llm_response

if __name__ == "__main__":
    print( "Reply = ", query_rag("What does the fox jump over?") )