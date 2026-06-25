#!/bin/bash
docker exec my-local-rag-ollama-1 ollama pull nomic-embed-text
docker exec my-local-rag-ollama-1 ollama pull llama3.2:1b
docker exec my-local-rag-ollama-1 ollama pull qwen2.5:0.5b
docker exec my-local-rag-ollama-1 ollama pull phi3:3.8b