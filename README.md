# *This project has been created as part of the 42 curriculum by imelero-.*

# RAG Against the Machine

## Description

**RAG Against the Machine** is an offline Retrieval-Augmented Generation (RAG) system developed as part of the 42 curriculum. The objective of the project is to answer technical questions about the provided **vLLM** repository by combining lexical retrieval with a lightweight local language model.

The pipeline follows the four classical RAG stages:

1. Indexing
2. Retrieval
3. Augmentation
4. Generation

The system indexes the supplied codebase, retrieves the most relevant source snippets using BM25, injects them into the context of a local Qwen model, and produces grounded answers while preserving the original source locations.

The project was designed to satisfy the mandatory requirements of the subject, including:

- Offline execution
- Python Fire CLI
- Pydantic data models
- BM25 lexical retrieval
- Qwen/Qwen3-0.6B answer generation
- Recall@k evaluation

---

# System Architecture

```text
                Raw vLLM repository
                        │
                        ▼
              Document ingestion
                        │
                        ▼
           Syntax-aware chunking
                        │
                        ▼
             BM25 index creation
                        │
                        ▼
             Top-k document retrieval
                        │
                        ▼
             Prompt construction
                        │
                        ▼
             Qwen/Qwen3-0.6B
                        │
                        ▼
              Structured JSON output
                        │
                        ▼
                Recall@k evaluation
```

The pipeline is composed of independent modules responsible for parsing, indexing, retrieval, generation and evaluation.

---

# Chunking Strategy

Two different chunking strategies are implemented as required by the project.

## Python files

Python source code is segmented while preserving semantic boundaries. Instead of splitting every fixed number of characters, the parser searches backwards for the nearest `def` or `class` declaration whenever the configured chunk size is reached. This keeps complete functions and classes inside the same chunk whenever possible.

## Markdown files

Markdown documentation is divided using section headers (`#`, `##`, ...). Large sections are recursively divided while respecting the maximum chunk length.

The maximum chunk size is configurable through the CLI (`--max_chunk_size`) and defaults to **2000 characters**, matching the project requirements.

---

# Retrieval Method

Retrieval is performed using **BM25s**, an optimized implementation of the classical BM25 ranking algorithm.

Each document chunk is tokenized during indexing and stored inside a persistent lexical index.

When a query is received:

1. The query is tokenized.
2. BM25 scores every indexed chunk.
3. The Top-k ranked chunks are returned.
4. Their source locations are preserved for answer generation.

BM25 was selected because it is lightweight, deterministic, fast and satisfies the mandatory requirement of implementing a lexical retrieval method.

---

# Answer Generation

The retrieved chunks are inserted into a structured prompt and passed to the mandatory local language model:

- Qwen/Qwen3-0.6B

Generation is configured to be deterministic by disabling sampling (`do_sample=False`), ensuring reproducible answers during evaluation.

The final output follows the provided Pydantic models and contains:

- question id
- question
- retrieved sources
- generated answer

---

# Performance Analysis

The implemented system satisfies the mandatory Recall@5 requirements.

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---------|---------:|---------:|---------:|----------:|
| Documentation | 60.0% | 77.0% | **85.0%** | 88.0% |
| Code | 31.3% | 47.5% | **53.5%** | 62.6% |

Performance observations:

- Repository indexing: approximately **5.4 seconds**
- Retrieval throughput: below **10 ms per query**
- Memory usage remains low thanks to sparse BM25 structures.

---

# Design Decisions

Several implementation decisions were made to maximize retrieval quality while keeping the system lightweight.

- Separate chunking strategies for Python and Markdown files.
- BM25 lexical retrieval instead of dense embeddings.
- Deterministic greedy decoding.
- Pydantic validation between every pipeline stage.
- Modular architecture separating parsing, indexing, retrieval, generation and evaluation.
- In-memory chunk lookup to avoid unnecessary disk reads.

---

# Challenges Faced

## Chunk boundaries

Naive fixed-size chunking frequently split functions and classes, reducing retrieval quality. This was solved using syntax-aware chunking.

## Retrieval quality

Different BM25 parameters and chunk sizes were evaluated before achieving the required Recall@5 values.

## Hallucinations

The language model occasionally produced unsupported information. Restricting generation to retrieved context and using deterministic decoding reduced this behaviour.

## Performance

The retrieval pipeline had to remain below the execution limits defined by the subject while indexing the entire repository efficiently.

---

# Project Structure

```text
rag-against-the-machine/
├── data/
│   ├── datasets/
│   ├── raw/
│   ├── processed/
│   └── output/
├── src/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── ingest.py
│   ├── models.py
│   ├── utils.py
├── Makefile
├── pyproject.toml
└── README.md
└── uv.lock
```

---

# Instructions

## Installation

```bash
make install
```

Dependencies are managed with **uv**.

The Qwen model is automatically downloaded by Hugging Face Transformers during its first execution.

## Build the index

```bash
uv run python -m src index --max_chunk_size 2000
```

## Search a single query

```bash
uv run python -m src search "How to configure OpenAI server?" --k 10
```

## Search a dataset

```bash
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results/UnansweredQuestions
```

## Answer a single query

```bash
uv run python -m src answer "What is the default timeout value for vLLM RPC operations?" --k 10
```

## Answer a dataset

```bash
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

## Evaluate

```bash
uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

---

# Example Usage

Generate an answer for a single question:

```bash
uv run python -m src answer "What is the default timeout value for vLLM RPC operations?" --k 10
```

The pipeline retrieves the most relevant source locations, builds a prompt using those snippets and generates a grounded answer together with the retrieved sources.

---

# Resources

## Documentation

- Python Documentation
- Hugging Face Transformers Documentation
- PyTorch Documentation
- Pydantic Documentation
- Python Fire Documentation
- BM25 documentation and literature
- vLLM Documentation

## AI Usage

Gemini and ChatGPT were used as learning and documentation tools.

They were mainly used to:
- better understand concepts related to the vLLM repository;
- clarify theoretical aspects of Retrieval-Augmented Generation;
- improve the quality of the project documentation.

AI was **not** used to design the system architecture, implement the retrieval pipeline, write the project code, or make technical decisions. All implementation, debugging, tuning and design choices were carried out by the project author.
