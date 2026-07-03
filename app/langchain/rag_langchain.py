from .clients_langchain import OLLAMA_HOST, vectorstore

from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# from langchain_core.runnables import RunnableLambda

retriever = vectorstore.as_retriever(search_kwargs={
    "k": 1
})

llm = ChatOllama(
    model="phi3:3.8b",
    temperature=0,
    base_url=OLLAMA_HOST
)

prompt_template = ChatPromptTemplate.from_template("""
You are a precise information extraction assistant.
Rules:
1. Answer ONLY using the information in the provided text.
2. Do NOT use prior knowledge or assumptions.
3. Return the shortest accurate answer found in the text.
4. If the answer is not in the text, return exactly: NOT FOUND
5. Never explain, justify, or add context to your answer."
 
Text:
{context}
 
Question: {question}
""")

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
#    | RunnableLambda(lambda x: print(x) or x)
    | prompt_template
    | llm
    | StrOutputParser()
)

def query_rag_langchain(question: str) -> str:
    return chain.invoke(question)

####
# EMBEDDINGS ARE NOT COMPATIBLE: Ollama normalizes embeddings
####
# # bare-bones RAG uses ollama.Client which calls /api/embeddings
# # /api/embeddings (old) → returns raw vectors from the model
# from clients import ollama_client
# v1 = ollama_client.embeddings(model="nomic-embed-text", prompt="test").embedding
# print(v1[:5])

# # OllamaEmbeddings from langchain-ollama calls the newer /api/embed
# # /api/embed (new) → returns L2-normalized vectors by default
# v2 = OllamaEmbeddings(model="nomic-embed-text").embed_query("test")
# print(v2[:5])  # these differ, that's your bug

# # Same model weights = same raw embeddings. The real difference is post-processing.
# # Ollama applies it differently between the two endpoints:
# # - ollama_client.embeddings > /api/embeddings (old) → returns raw vectors from the model
# # - OllamaEmbeddings > /api/embed (new) → returns L2-normalized vectors by default
# import numpy as np
# print("v1 magnitude:", np.linalg.norm(v1))   # raw
# print("v2 magnitude:", np.linalg.norm(v2))   # L2 normalized → should be ~1.0