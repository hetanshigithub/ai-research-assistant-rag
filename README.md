# AI Research Assistant with Local RAG Pipeline

A local, API-free research assistant that answers questions from a folder of documents using Retrieval-Augmented Generation (RAG). The project is designed to demonstrate practical AI engineering skills: document ingestion, chunking, embeddings, semantic retrieval, evidence-grounded answer generation, confidence scoring, and clean CLI output.

## Project Overview

This assistant loads TXT/PDF documents from `data/`, chunks them into searchable passages, embeds the chunks with `sentence-transformers`, retrieves the most relevant evidence with cosine similarity, and generates a structured answer using only the retrieved context.

It is intentionally local and lightweight. There are no paid APIs, no hosted vector database, and no unnecessary web framework. The goal is to show the core mechanics behind real RAG products in a way that is easy to inspect, explain, and extend.

## Problem It Solves

Research-heavy workflows often require reading across notes, reports, filings, policy documents, and internal knowledge bases. A normal chatbot can produce fluent but unsupported answers. This project improves that workflow by:

- Searching a trusted local document set
- Showing the top retrieved evidence before the answer
- Including source files, chunk IDs, and similarity scores
- Producing a structured answer with risks and limitations
- Saving each run to JSON for review and debugging

The included sample corpus focuses on AI in finance, RAG pipelines, LLM evaluation, research automation, and risks of AI in investment workflows.

## What Is RAG?

Retrieval-Augmented Generation is an architecture where a system retrieves relevant context from documents before generating an answer. Instead of answering from memory, the assistant grounds its response in evidence from the local knowledge base. This makes answers more transparent, auditable, and domain-specific.

## Architecture Flow

```text
User Query
   |
   v
main.py
   |
   +--> data_loader.py
   |       Loads TXT/PDF documents from data/
   |
   +--> chunking.py
   |       Splits documents into overlapping word chunks
   |
   +--> embeddings.py
   |       Creates local sentence-transformer embeddings
   |
   +--> retriever.py
   |       Computes cosine similarity and returns top-k evidence
   |
   +--> generator.py
   |       Builds a structured answer from retrieved chunks
   |
   v
outputs/results.json
```

## Folder Structure

```text
ai-research-assistant-rag/
|-- main.py
|-- requirements.txt
|-- README.md
|-- data/
|   |-- ai_in_finance.txt
|   |-- ai_investment_risks.txt
|   |-- llm_evaluation_playbook.txt
|   |-- rag_pipeline_notes.txt
|   |-- research_automation.txt
|-- src/
|   |-- data_loader.py
|   |-- chunking.py
|   |-- embeddings.py
|   |-- retriever.py
|   |-- generator.py
|-- outputs/
|   |-- .gitkeep
|-- tests/
|   |-- test_chunking.py
|   |-- test_generator.py
```

## Installation

```bash
cd /Users/hetanshiii/Documents/Playground/ai-research-assistant-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first run may download the local embedding model from Hugging Face. After the model is cached, future runs are faster.

## How To Run

Basic query:

```bash
python main.py "impact of AI in finance"
```

Show the top 3 retrieved chunks:

```bash
python main.py "impact of AI in finance" --top_k 3 --show_chunks true
```

Save output to `outputs/results.json`:

```bash
python main.py "risks of AI in investment workflows" --top_k 4 --show_chunks true --save_json true
```

Hide retrieved evidence in the terminal:

```bash
python main.py "how should RAG systems be evaluated?" --show_chunks false
```

Use a custom data folder:

```bash
python main.py "summarize the main risks" --data-dir data --top_k 5
```

The project also supports the hyphenated alias:

```bash
python main.py "research automation use cases" --top-k 3
```

## Output Format

The terminal answer is always structured as:

```text
Summary
Key Insights
Risks / Limitations
Supporting Evidence
Confidence Score
```

Example:

```text
========================================================================================
Top Retrieved Evidence
========================================================================================

Rank: 1
Similarity Score: 0.641
Source File: ai_investment_risks.txt
Chunk ID: 9
Preview: AI can improve investment workflows, but it also introduces risks...

========================================================================================
STRUCTURED ANSWER
========================================================================================

Summary:
Based on the retrieved evidence, AI can improve investment workflows, but it also introduces risks.

Key Insights:
- Financial firms should treat AI outputs as decision-support signals, not authoritative recommendations.
- RAG systems can fail when retrieval quality is poor.
- Local systems reduce exposure to external APIs, but teams still need access controls.

Risks / Limitations:
- The most important risk is overreliance: users may treat fluent model output as verified analysis.
- RAG systems can also fail when retrieval quality is poor.

Supporting Evidence:
- ai_investment_risks.txt | chunk 9 | score=0.641: AI can improve investment workflows...

Confidence Score:
0.734
```

## JSON Logging

When `--save_json true` is enabled, each run is appended to:

```text
outputs/results.json
```

Each run stores:

- Timestamp
- Query
- Retrieved source files
- Chunk IDs
- Similarity scores
- Final answer
- Confidence score
- Basic run metadata

## Confidence Scoring

The confidence score is deterministic and heuristic-based. It uses:

- Average similarity score of the top retrieved chunks
- Keyword overlap between the query and retrieved context
- Whether supporting evidence exists

It is not random, and it is not meant to be a perfect truth score. It is a practical signal for debugging answer quality and retrieval strength.

## Limitations

- The answer generator is rule-based to avoid paid APIs.
- It summarizes retrieved evidence but does not have the reasoning ability of a larger local LLM.
- The retriever is in-memory, so very large corpora would need a vector database.
- PDF extraction quality depends on the structure of the PDF.
- Confidence scoring is a simple heuristic, not a formal evaluation benchmark.

## Future Improvements

- Add FAISS or Chroma for persistent vector search
- Add query benchmarking with expected evidence sources
- Add source-level citations inside each generated sentence
- Add a lightweight Streamlit UI for side-by-side evidence review
- Support Markdown and CSV documents
- Add reranking for better top-k retrieval quality
- Add optional local LLM generation through Ollama or llama.cpp

## Why This Project Is Relevant For AI/ML Internships

This project demonstrates skills that applied AI teams care about:

- Building an end-to-end RAG pipeline
- Working with local embeddings and cosine similarity
- Designing modular AI system components
- Handling document ingestion and chunking
- Producing evidence-grounded outputs
- Adding confidence scoring and JSON logging
- Keeping the system practical, local, and explainable

It is realistic enough to discuss in interviews without pretending to be a full enterprise platform. The project shows that you understand the engineering behind modern AI assistants, not just prompting.
