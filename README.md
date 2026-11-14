# Automotive RAG System

Gemini-first retrieval-augmented generation system for automotive knowledge, built as a modular Python service with evaluation, typed configuration, and a FastAPI layer.

## What This Project Does

This repository turns a single-file RAG prototype into a cleaner architecture for:

- Question answering over automotive documents
- PDF ingestion and chunking
- Vector search with Chroma
- Gemini-powered generation with local fallback support
- RAGAS-based evaluation and quality gates
- FastAPI endpoints for integration and demos
- MinerU-backed PDF parsing for cleaner document extraction before chunking

The current codebase is structured to be easy to extend into LangGraph multi-agent orchestration, MLflow tracking, Dockerized deployment, and CI/CD.

## Why It Stands Out

- Gemini stays the primary model, as requested.
- Configuration is centralized with Pydantic settings instead of hardcoded constants.
- Retrieval, embeddings, vector store, API, and evaluation are separated into focused modules.
- The repo includes a baseline 10-Q evaluation path and an automotive corpus path for the next phase.
- The architecture is recruiter-friendly because it shows practical engineering judgment, not just a demo script.

## Tech Stack

- Python 3.11+
- LangChain ecosystem
- Google Gemini
- Hugging Face embeddings
- Chroma vector database
- FastAPI + Uvicorn
- RAGAS for evaluation
- Pydantic Settings
- MinerU for PDF parsing and markdown/json extraction

## Project Structure

```text
src/
  config/        Settings and environment handling
  data/          Document loading and chunking
  rag/           Embeddings and vector store management
  agents/        LLM factory and RAG chain
  api/           FastAPI app and routes
  eval/          Datasets and evaluation helpers

main.py          CLI entry point
server.py        FastAPI entry point
scripts/         Setup helpers
data/            Local-only documents and corpora
```

### Automotive corpus layout

Put raw documents here:

```text
data/automotive_corpus/incoming_pdfs/
```

MinerU output will be written here:

```text
data/automotive_corpus/parsed/
```

Recommended subfolders for the source corpus:

```text
data/automotive_corpus/
  incoming_pdfs/
  parsed/
  vehicle_specs/
  safety_reports/
  maintenance_guides/
  regulations/
```

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Add your Gemini API key to `.env`.
4. Place automotive PDFs in `data/automotive_corpus/incoming_pdfs/`.
5. Run `python scripts/setup.py` to verify the environment.
6. Run `python main.py` for the CLI experience and evaluation.
7. Run `python server.py` to start the API.

### MinerU parsing flow

1. Put a PDF into `data/automotive_corpus/incoming_pdfs/`.
2. `DocumentLoader` calls MinerU first.
3. MinerU converts the file into Markdown or JSON in `data/automotive_corpus/parsed/`.
4. The parsed text is chunked and indexed into Chroma.
5. If MinerU is unavailable, the loader falls back to `PyPDFLoader`.

## API

- `GET /health` - basic service health
- `POST /api/v1/chat` - ask a question and get an answer

## Automotive Data Sources to Use Next

Good first-phase data sources for the automotive corpus:

- Kaggle vehicle and EV datasets
- NHTSA recalls and safety data
- SEC 10-K filings for automotive companies
- GovInfo / eCFR / CFR regulatory PDFs
- Manufacturer manuals and owner guides published publicly by OEMs
- Open transport and safety portals from government agencies

The repo is set up so these sources can be converted into a structured corpus under `data/automotive_corpus/` locally.

### Best places to find PDFs

- NHTSA recall and safety publications
- GovInfo and eCFR regulatory PDFs
- OEM owner manuals and technical guides from public manufacturer sites
- SEC filings for automotive companies that include annual reports and sustainability reports
- Kaggle or Hugging Face datasets when the source is already distributed as PDF or can be exported to PDF

## Roadmap

- Phase 1: modularize and stabilize the current RAG pipeline
- Phase 2: ingest automotive documents and compare against the current baseline
- Phase 3: add LangGraph multi-agent orchestration
- Phase 4: add observability, tracking, and deployment automation

## For Recruiters

This project shows:

- Real refactoring from monolith to modular design
- Typed configuration and clean boundaries
- Practical retrieval and evaluation work
- AI system design with production concerns in mind
- Clear path from prototype to deployable service