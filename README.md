# Autonomous Research Agent

A local-first autonomous research assistant that can ingest academic papers, semantically retrieve relevant information, and generate grounded answers using Retrieval-Augmented Generation (RAG).

Built with Docling, Qdrant, Sentence Transformers, LangGraph, and Ollama.

---

# Features

- PDF ingestion and parsing
- Layout-aware document extraction using Docling
- Hybrid semantic chunking
- Dense vector retrieval with Qdrant
- Cross-encoder reranking
- Local LLM inference using Ollama
- LangGraph orchestration pipeline
- Grounding verification for hallucination reduction
- Dynamic querying over any research paper
- Fully local pipeline (no OpenAI API required)

---

## Tech Stack

- Python
- Docling
- Sentence Transformers
- Qdrant
- LangGraph
- Ollama
- Phi3
- HuggingFace Transformers

---

## Project Architecture

```text
PDF
 ↓
Docling Parsing
 ↓
Hybrid Semantic Chunking
 ↓
Embedding Generation
 ↓
Qdrant Vector Database
 ↓
Dense Retrieval
 ↓
Cross-Encoder Reranking
 ↓
LLM Generation (Ollama)
 ↓
Grounding Verification
 ↓
Final Answer

