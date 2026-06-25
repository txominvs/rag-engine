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
        options={"temperature": 0},
        messages=
        [
            {
                "role": "user",
                "content": f"""Extract the exact answer verbatim from the text.
If not present, return "NOT FOUND".

Text: {context}
Question: {question}
Answer:"""
            }
        ]
    ).message.content

    return llm_response

if __name__ == "__main__":
    while user_input := input("Question?"):
        print( "Reply = ", query_rag(user_input) )