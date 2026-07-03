#!/bin/bash
docker exec rag-engine-ollama-1 ollama pull nomic-embed-text
docker exec rag-engine-ollama-1 ollama pull llama3.2:1b
docker exec rag-engine-ollama-1 ollama pull qwen2.5:0.5b
docker exec rag-engine-ollama-1 ollama pull phi3:3.8b