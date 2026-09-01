# Retrieval-Augmented Generation (RAG)

## Overview

Retrieval-Augmented Generation (RAG) is an AI framework that combines information retrieval with text generation. It addresses the limitations of large language models (LLMs) by grounding their responses in external knowledge sources.

## How RAG Works

1. **Indexing**: Documents are split into chunks, embedded into vectors, and stored in a vector database.
2. **Retrieval**: When a user asks a question, it's embedded and similar chunks are retrieved from the index.
3. **Generation**: The retrieved chunks are provided as context to an LLM, which generates a grounded response.

## Benefits

- **Factual grounding**: Reduces hallucinations by providing source material
- **Updatable knowledge**: Can add new documents without retraining the model
- **Citations**: Can reference specific source documents
- **Domain adaptation**: Works with private/proprietary data

## Challenges

- **Chunking strategy**: How to split documents optimally
- **Embedding quality**: Choice of embedding model affects retrieval
- **Context window**: Limited by LLM context size
- **Latency**: Retrieval adds overhead to generation

## Evaluation Metrics

- **Retrieval**: Recall@K, MRR, NDCG
- **Generation**: Faithfulness, answer relevance, citation accuracy
- **End-to-end**: User satisfaction, task completion rate