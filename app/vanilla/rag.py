from .clients import ollama_client, chroma_collection

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

    context = "\n---\n".join(documents)

    print(f"{context = }")

    llm_response = ollama_client.chat(
        model="phi3:3.8b",
        options={"temperature": 0},
        messages=
        [
        {
            "role": "system",
            "content": (
                "You are a precise information extraction assistant.\n"
                "Rules:\n"
                "1. Answer ONLY using the information in the provided text.\n"
                "2. Do NOT use prior knowledge or assumptions.\n"
                "3. Return the shortest accurate answer found in the text.\n"
                "4. If the answer is not in the text, return exactly: NOT FOUND\n"
                "5. Never explain, justify, or add context to your answer."
            )
        },
        {
            "role": "user",
            "content": f"Text:\n{context}\n\nQuestion: {question}"
        }
        ]
    ).message.content

    return llm_response

if __name__ == "__main__":
    while user_input := input("Question?"):
        print( "Reply = ", query_rag(user_input) )