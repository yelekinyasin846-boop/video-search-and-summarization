# Video Search and Summarization

A fork of [NVIDIA AI Blueprints: Video Search and Summarization](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization).

This blueprint enables intelligent video search and summarization using NVIDIA AI technologies, including multimodal embeddings, vision-language models, and retrieval-augmented generation (RAG).

## Overview

This project provides a pipeline for:
- **Ingesting** video files and extracting frames/metadata
- **Indexing** video content using multimodal embeddings (visual + text)
- **Searching** across video libraries using natural language queries
- **Summarizing** video segments using vision-language models

## Features

- 🎥 Multi-format video ingestion (MP4, AVI, MOV, MKV)
- 🔍 Natural language semantic search over video content
- 📝 AI-powered video summarization
- 🖼️ Frame-level visual understanding
- 🗣️ Audio transcription and indexing
- 📊 REST API for integration

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Video Ingestion                   │
│  (Frame Extraction + Audio Transcription + OCR)     │
└───────────────────┬─────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────┐
│              Embedding & Indexing                   │
│     (NVIDIA Multimodal Embeddings + Vector DB)      │
└───────────────────┬─────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────┐
│            Search & Summarization API               │
│        (RAG Pipeline + VLM Summarization)           │
└─────────────────────────────────────────────────────┘
```

## Prerequisites

- Python 3.10+
- Docker & Docker Compose
- NVIDIA GPU (recommended: A100, H100, or RTX 4090)
- NVIDIA AI Enterprise license (for NIM microservices)
- [NGC API Key](https://ngc.nvidia.com/)

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/video-search-and-summarization.git
cd video-search-and-summarization
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your NGC API key and configuration
```

### 3. Launch Services

```bash
docker compose up -d
```

### 4. Ingest Videos

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@/path/to/video.mp4"
```

### 5. Search

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "person wearing red jacket", "top_k": 5}'
```

## Configuration

See [docs/configuration.md](docs/configuration.md) for full configuration reference.

Key environment variables:

| Variable | Description | Default |
|---|---|---|
| `NGC_API_KEY` | NVIDIA NGC API key | required |
| `EMBEDDING_MODEL` | Multimodal embedding model | `nvidia/nvclip` |
| `VLM_MODEL` | Vision-language model for summarization | `nvidia/llama-3.2-90b-vision-instruct` |
| `VECTOR_DB_URL` | Vector database connection URL | `http://milvus:19530` |
| `TOP_K_DEFAULT` | Default number of search results returned | `5` |

> **Personal note:** I'm using `TOP_K_DEFAULT=10` in my local `.env` since I find more results useful when exploring unfamiliar video libraries.
