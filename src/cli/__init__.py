"""
CLI entry point for the RAG pipeline.
"""

import sys
from pathlib import Path

import click

from ..rag.config import config, setup_logging
from ..rag.ingestion import DocumentChunker
from ..rag.embeddings import get_embedding_provider
from ..rag.index import BruteForceIndex, HNSWIndex
from ..rag.query import QueryEngine


def _load_index(index_path: Path):
    """Load index (BruteForce or HNSW) from disk."""
    import json
    config_path = index_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        config_data = json.load(f)

    index_type = config_data.get("index_type", "brute_force")

    if index_type == "hnsw":
        return HNSWIndex.load(index_path)
    else:
        return BruteForceIndex.load(index_path)


def _load_chunks(index_path: Path):
    """Load chunks from index metadata."""
    import json
    chunks_path = index_path / "chunks.json"
    if not chunks_path.exists():
        return []

    with open(chunks_path, "r") as f:
        chunks_data = json.load(f)

    from ..rag.ingestion import Chunk
    chunks = []
    for data in chunks_data:
        chunks.append(Chunk(
            text=data["text"],
            source_file=data["source_file"],
            start_char=data["start_char"],
            end_char=data["end_char"],
            chunk_index=data["chunk_index"],
            metadata=data.get("metadata", {}),
        ))
    return chunks


@click.group()
@click.option("--log-level", default=None, help="Log level (DEBUG, INFO, WARNING, ERROR)")
@click.pass_context
def cli(ctx, log_level):
    """RAG Pipeline - Retrieval-Augmented Generation with custom HNSW index."""
    ctx.ensure_object(dict)
    if log_level:
        config.log_level = log_level
    setup_logging(config)


@cli.command()
@click.argument("data_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--index-path", type=click.Path(path_type=Path), default="./data/index", help="Index output directory")
@click.option("--glob", default="**/*.txt", help="File glob pattern")
@click.option("--chunk-size", default=None, type=int, help="Chunk size (overrides config)")
@click.option("--overlap", default=None, type=int, help="Chunk overlap (overrides config)")
@click.option("--index-type", type=click.Choice(["brute_force", "hnsw"]), default=None, help="Index type (overrides config)")
@click.pass_context
def ingest(ctx, data_dir, index_path, glob, chunk_size, overlap, index_type):
    """Ingest documents and build the vector index."""
    index_type = index_type or config.index_type
    click.echo(f"Ingesting documents from {data_dir} (index: {index_type})...")

    # Initialize components
    chunker = DocumentChunker(chunk_size=chunk_size, overlap=overlap)
    embedder = get_embedding_provider()

    if index_type == "hnsw":
        index = HNSWIndex(
            dimension=embedder.dimension,
            M=16,
            ef_construction=200,
            ef_search=50,
        )
    else:
        index = BruteForceIndex(dimension=embedder.dimension)

    # Chunk all files
    all_chunks = chunker.chunk_directory(data_dir, glob_pattern=glob)

    if not all_chunks:
        click.echo("No chunks generated. Check your data directory and glob pattern.")
        sys.exit(1)

    click.echo(f"Generated {len(all_chunks)} chunks. Generating embeddings...")

    # Generate embeddings in batches
    batch_size = 32
    texts = [chunk.text for chunk in all_chunks]

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_chunks = all_chunks[i:i + batch_size]
        embeddings = embedder.embed(batch_texts)
        index.add(embeddings, batch_chunks)
        click.echo(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)} chunks")

    # Save index
    index.save(index_path)
    click.echo(f"Index saved to {index_path} ({index.size} vectors)")


@cli.command()
@click.argument("query")
@click.option("--index-path", type=click.Path(exists=True, path_type=Path), default="./data/index", help="Index directory")
@click.option("-k", "--top-k", default=10, help="Number of results")
@click.option("--hybrid/--no-hybrid", default=None, help="Use hybrid search (BM25 + Dense)")
@click.option("--rerank/--no-rerank", default=None, help="Use cross-encoder reranker")
@click.option("--hyde/--no-hyde", default=None, help="Use HyDE query expansion")
@click.option("--multi-query/--no-multi-query", default=None, help="Use multi-query expansion")
@click.option("--compound/--no-compound", default=True, help="Decompose compound queries")
@click.pass_context
def search(ctx, query, index_path, top_k, hybrid, rerank, hyde, multi_query, compound):
    """Search the index with a query."""
    click.echo(f"Searching for: {query}")

    # Load index and chunks
    index = _load_index(index_path)
    chunks = _load_chunks(index_path)

    # Override config from CLI flags
    if hybrid is not None:
        config.use_hybrid_search = hybrid
    if rerank is not None:
        config.use_reranker = rerank
    if hyde is not None:
        config.use_hyde = hyde
    if multi_query is not None:
        config.use_multi_query = multi_query

    # Create engine with chunks for hybrid search
    engine = QueryEngine(index, chunks=chunks)

    # Search
    result = engine.search(query, k=top_k, decompose_compound_queries=compound)

    # Display results
    click.echo(f"\nFound {len(result.results)} results:\n")
    for r in result.results:
        click.echo(f"  Rank {r.rank} (score: {r.score:.4f})")
        click.echo(f"    Source: {r.chunk.source_file}")
        click.echo(f"    Text: {r.chunk.text[:200]}...")
        click.echo()


@cli.command()
@click.option("--index-path", type=click.Path(exists=True, path_type=Path), default="./data/index", help="Index directory")
@click.pass_context
def stats(ctx, index_path):
    """Show index statistics."""
    index = _load_index(index_path)
    click.echo(f"Index type: {type(index).__name__}")
    click.echo(f"Dimension: {index.dimension}")
    click.echo(f"Vectors: {index.size}")

    if hasattr(index, 'get_stats'):
        stats = index.get_stats()
        click.echo(f"Max Layer: {stats.get('max_layer', 'N/A')}")
        click.echo(f"Layers: {stats.get('num_layers', 'N/A')}")
        click.echo(f"Nodes per Layer: {stats.get('nodes_per_layer', 'N/A')}")
        click.echo(f"Avg Links/Node: {stats.get('avg_links_per_node', 'N/A'):.1f}")


@cli.command()
@click.argument("query")
@click.option("--index-path", type=click.Path(exists=True, path_type=Path), default="./data/index", help="Index directory")
@click.option("-k", "--top-k", default=10, help="Number of chunks to retrieve")
@click.option("--max-tokens", default=1024, help="Max tokens for generation")
@click.option("--temperature", default=0.1, help="LLM temperature")
@click.option("--hybrid/--no-hybrid", default=None, help="Use hybrid search")
@click.option("--rerank/--no-rerank", default=None, help="Use cross-encoder reranker")
@click.option("--hyde/--no-hyde", default=None, help="Use HyDE query expansion")
@click.option("--multi-query/--no-multi-query", default=None, help="Use multi-query expansion")
@click.pass_context
def ask(ctx, query, index_path, top_k, max_tokens, temperature, hybrid, rerank, hyde, multi_query):
    """Ask a question and get a grounded answer (full RAG)."""
    click.echo(f"Question: {query}")

    # Load index and chunks
    index = _load_index(index_path)
    chunks = _load_chunks(index_path)

    # Override config from CLI flags
    if hybrid is not None:
        config.use_hybrid_search = hybrid
    if rerank is not None:
        config.use_reranker = rerank
    if hyde is not None:
        config.use_hyde = hyde
    if multi_query is not None:
        config.use_multi_query = multi_query

    # Create engine with chunks for hybrid search
    engine = QueryEngine(index, chunks=chunks)

    # Generate answer
    from ..rag.llm import get_llm_provider
    llm = get_llm_provider("dummy")  # Use dummy for CLI without API keys
    engine._llm_provider = llm

    result = engine.generate(query, k=top_k, max_tokens=max_tokens, temperature=temperature)

    click.echo(f"\nAnswer:\n{result.answer}\n")

    if result.citations:
        click.echo("Citations:")
        for c in result.citations:
            click.echo(f"  [{c.source_file}, chunk {c.chunk_index}] (score: {c.score:.3f})")
            click.echo(f"    {c.text_snippet}")


def main():
    """Entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()