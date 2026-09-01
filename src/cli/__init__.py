"""
CLI entry point for the RAG pipeline.
"""

import sys
from pathlib import Path

import click

from ..rag.config import config, setup_logging
from ..rag.ingestion import DocumentChunker
from ..rag.embeddings import get_embedding_provider
from ..rag.index import BruteForceIndex
from ..rag.query import QueryEngine


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
@click.pass_context
def ingest(ctx, data_dir, index_path, glob, chunk_size, overlap):
    """Ingest documents and build the vector index."""
    click.echo(f"Ingesting documents from {data_dir}...")

    # Initialize components
    chunker = DocumentChunker(chunk_size=chunk_size, overlap=overlap)
    embedder = get_embedding_provider()
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
@click.pass_context
def search(ctx, query, index_path, top_k):
    """Search the index with a query."""
    click.echo(f"Searching for: {query}")

    # Load index
    index = BruteForceIndex.load(index_path)
    engine = QueryEngine(index)

    # Search
    result = engine.search(query, k=top_k)

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
    index = BruteForceIndex.load(index_path)
    click.echo(f"Index type: BruteForceIndex")
    click.echo(f"Dimension: {index.dimension}")
    click.echo(f"Vectors: {index.size}")


def main():
    """Entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()