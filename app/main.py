from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.vanilla.rag import query_rag
from app.langchain.rag_langchain import query_rag_langchain

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/vanilla")
async def query_vanilla(question: str):
    return {"answer": query_rag(question)}

@app.post("/langchain")
async def query_langchain(question: str):
    return {"answer": query_rag_langchain(question)}

# Option 1: Use a different port for local development
# source .venv/bin/activate
# python -m uvicorn app.main:app --reload --port 8001
