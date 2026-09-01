#!/usr/bin/env python3
"""
HNSW Benchmark Script

Compares HNSWIndex against BruteForceIndex on synthetic data.
Measures recall@K, search speedup, and build time.
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag.index import BruteForceIndex, HNSWIndex
from rag.ingestion import Chunk


def generate_synthetic_data(
    n_vectors: int = 2000,
    dimension: int = 384,
    seed: int = 42,
) -> Tuple[np.ndarray, List[Chunk]]:
    """
    Generate synthetic dataset with clustered structure.

    Creates vectors with some cluster structure to make the task realistic.
    """
    rng = np.random.default_rng(seed)

    # Create cluster centers
    n_clusters = min(20, n_vectors // 50)
    cluster_centers = rng.normal(0, 1, size=(n_clusters, dimension))
    cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1, keepdims=True)

    # Assign vectors to clusters
    vectors = []
    chunks = []

    for i in range(n_vectors):
        cluster_idx = i % n_clusters
        center = cluster_centers[cluster_idx]

        # Add noise around cluster center
        noise = rng.normal(0, 0.3, size=dimension)
        vector = center + noise
        vector = vector / np.linalg.norm(vector)

        vectors.append(vector)
        chunks.append(Chunk(
            text=f"Document {i} from cluster {cluster_idx}",
            source_file=f"synthetic_cluster_{cluster_idx}.txt",
            start_char=0,
            end_char=50,
            chunk_index=i,
            metadata={"cluster": cluster_idx, "doc_id": i},
        ))

    return np.stack(vectors).astype(np.float32), chunks


def compute_recall(
    brute_results: List[List[int]],
    hnsw_results: List[List[int]],
    k: int,
) -> float:
    """
    Compute recall@K.

    Args:
        brute_results: List of lists of true neighbor indices (from brute force)
        hnsw_results: List of lists of found neighbor indices (from HNSW)
        k: Number of neighbors to consider

    Returns:
        Recall@K as float in [0, 1]
    """
    total_recall = 0.0
    for brute_nbrs, hnsw_nbrs in zip(brute_results, hnsw_results):
        brute_set = set(brute_nbrs[:k])
        hnsw_set = set(hnsw_nbrs[:k])
        if len(brute_set) > 0:
            total_recall += len(brute_set & hnsw_set) / len(brute_set)
    return total_recall / len(brute_results) if brute_results else 0.0


def run_benchmark(
    n_vectors: int = 2000,
    dimension: int = 384,
    n_queries: int = 100,
    k: int = 10,
    seed: int = 42,
    # HNSW parameters
    M: int = 16,
    ef_construction: int = 200,
    ef_search: int = 50,
) -> dict:
    """
    Run full benchmark comparing BruteForce vs HNSW.

    Returns dictionary with benchmark results.
    """
    print("=" * 70)
    print("HNSW vs Brute-Force Benchmark")
    print("=" * 70)
    print(f"\nDataset: {n_vectors} vectors, {dimension} dimensions")
    print(f"Queries: {n_queries}, k={k}")
    print(f"HNSW params: M={M}, ef_construction={ef_construction}, ef_search={ef_search}")

    # Generate data
    print("\n1. Generating synthetic data...")
    start = time.time()
    vectors, chunks = generate_synthetic_data(n_vectors, dimension, seed)
    gen_time = time.time() - start
    print(f"   Generated in {gen_time:.3f}s")

    # Build BruteForce index
    print("\n2. Building BruteForceIndex...")
    bf_index = BruteForceIndex(dimension=dimension)
    start = time.time()
    bf_index.add(vectors, chunks)
    bf_build_time = time.time() - start
    print(f"   Built in {bf_build_time:.3f}s ({bf_index.size} vectors)")

    # Build HNSW index
    print("\n3. Building HNSWIndex...")
    hnsw_index = HNSWIndex(
        dimension=dimension,
        M=M,
        ef_construction=ef_construction,
        ef_search=ef_search,
        seed=seed,
    )
    start = time.time()
    hnsw_index.add(vectors, chunks)
    hnsw_build_time = time.time() - start
    print(f"   Built in {hnsw_build_time:.3f}s ({hnsw_index.size} vectors)")

    # Print HNSW stats
    stats = hnsw_index.get_stats()
    print(f"\n   HNSW Structure:")
    print(f"     Layers: {stats['num_layers']} (max layer: {stats['max_layer']})")
    print(f"     Nodes per layer: {stats['nodes_per_layer']}")
    print(f"     Avg links/node: {stats['avg_links_per_node']:.1f}")
    print(f"     Entry point: {stats['entry_point']}")

    # Generate query vectors (subset of existing vectors + some noise)
    print("\n4. Generating queries...")
    query_indices = np.random.default_rng(seed + 1).choice(n_vectors, size=n_queries, replace=False)
    query_vectors = vectors[query_indices].copy()

    # Add small noise to queries to make it more realistic
    noise = np.random.default_rng(seed + 2).normal(0, 0.05, size=query_vectors.shape)
    query_vectors = query_vectors + noise
    query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)

    # Run BruteForce searches
    print("\n5. Running BruteForce searches...")
    bf_times = []
    bf_neighbors = []

    start = time.time()
    for qv in query_vectors:
        q_start = time.time()
        results = bf_index.search(qv, k=k)
        bf_times.append(time.time() - q_start)
        bf_neighbors.append([chunks.index(r.chunk) for r in results])
    bf_total_time = time.time() - start
    avg_bf_time = np.mean(bf_times) * 1000  # ms

    print(f"   Total: {bf_total_time:.3f}s, Avg per query: {avg_bf_time:.3f}ms")

    # Run HNSW searches
    print("\n6. Running HNSW searches...")
    hnsw_times = []
    hnsw_neighbors = []

    start = time.time()
    for qv in query_vectors:
        q_start = time.time()
        results = hnsw_index.search(qv, k=k)
        hnsw_times.append(time.time() - q_start)
        hnsw_neighbors.append([chunks.index(r.chunk) for r in results])
    hnsw_total_time = time.time() - start
    avg_hnsw_time = np.mean(hnsw_times) * 1000  # ms

    print(f"   Total: {hnsw_total_time:.3f}s, Avg per query: {avg_hnsw_time:.3f}ms")

    # Compute recall
    print("\n7. Computing Recall@K...")
    recalls = {}
    for k_val in [1, 5, 10, 20]:
        if k_val <= k:
            recall = compute_recall(bf_neighbors, hnsw_neighbors, k_val)
            recalls[k_val] = recall
            print(f"   Recall@{k_val}: {recall:.4f} ({recall*100:.2f}%)")

    # Compute speedup
    speedup = avg_bf_time / avg_hnsw_time if avg_hnsw_time > 0 else float('inf')

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"BruteForce build time:  {bf_build_time:.3f}s")
    print(f"HNSW build time:        {hnsw_build_time:.3f}s")
    print(f"Build time ratio:       {hnsw_build_time / bf_build_time:.2f}x (HNSW slower)")
    print(f"\nBruteForce avg query:   {avg_bf_time:.3f}ms")
    print(f"HNSW avg query:         {avg_hnsw_time:.3f}ms")
    print(f"Speedup:                {speedup:.2f}x faster")
    print(f"\nRecall@1:               {recalls.get(1, 0):.4f} ({recalls.get(1, 0)*100:.2f}%)")
    print(f"Recall@5:               {recalls.get(5, 0):.4f} ({recalls.get(5, 0)*100:.2f}%)")
    print(f"Recall@10:              {recalls.get(10, 0):.4f} ({recalls.get(10, 0)*100:.2f}%)")
    print(f"Recall@20:              {recalls.get(20, 0):.4f} ({recalls.get(20, 0)*100:.2f}%)")

    # Check if targets met
    print("\n" + "-" * 70)
    print("TARGET CHECKS:")
    target_recall = 0.90
    recall_10 = recalls.get(10, 0)
    if recall_10 >= target_recall:
        print(f"[PASS] Recall@10 >= 90%: PASS ({recall_10*100:.2f}%)")
    else:
        print(f"[FAIL] Recall@10 >= 90%: FAIL ({recall_10*100:.2f}%)")

    target_speedup = 5.0
    if speedup >= target_speedup:
        print(f"[PASS] Speedup >= 5x: PASS ({speedup:.2f}x)")
    else:
        print(f"[FAIL] Speedup >= 5x: FAIL ({speedup:.2f}x)")

    print("=" * 70)

    return {
        "n_vectors": n_vectors,
        "dimension": dimension,
        "n_queries": n_queries,
        "k": k,
        "hnsw_params": {
            "M": M,
            "ef_construction": ef_construction,
            "ef_search": ef_search,
        },
        "bf_build_time": bf_build_time,
        "hnsw_build_time": hnsw_build_time,
        "bf_avg_query_ms": avg_bf_time,
        "hnsw_avg_query_ms": avg_hnsw_time,
        "speedup": speedup,
        "recalls": recalls,
        "hnsw_stats": stats,
    }


def run_parameter_sweep():
    """Run benchmark with different HNSW parameters to find optimal settings."""
    print("\n" + "=" * 70)
    print("PARAMETER SWEEP")
    print("=" * 70)

    param_configs = [
        {"M": 8, "ef_construction": 100, "ef_search": 20},
        {"M": 16, "ef_construction": 200, "ef_search": 50},
        {"M": 32, "ef_construction": 200, "ef_search": 50},
        {"M": 16, "ef_construction": 400, "ef_search": 100},
        {"M": 16, "ef_construction": 200, "ef_search": 100},
        {"M": 16, "ef_construction": 200, "ef_search": 200},
    ]

    results = []
    for config in param_configs:
        print(f"\nTesting: M={config['M']}, ef_construction={config['ef_construction']}, ef_search={config['ef_search']}")
        try:
            result = run_benchmark(
                n_vectors=2000,
                dimension=384,
                n_queries=100,
                k=10,
                **config,
            )
            results.append(result)
        except Exception as e:
            print(f"   Error: {e}")
            results.append(None)

    # Summary table
    print("\n" + "=" * 70)
    print("PARAMETER SWEEP SUMMARY")
    print("=" * 70)
    print(f"{'M':>4} {'ef_c':>6} {'ef_s':>6} {'Build(s)':>8} {'Query(ms)':>9} {'Speedup':>8} {'R@10':>7}")
    print("-" * 70)
    for config, result in zip(param_configs, results):
        if result:
            print(f"{config['M']:>4} {config['ef_construction']:>6} {config['ef_search']:>6} "
                  f"{result['hnsw_build_time']:>8.3f} {result['hnsw_avg_query_ms']:>9.3f} "
                  f"{result['speedup']:>8.2f} {result['recalls'].get(10, 0)*100:>6.2f}%")
        else:
            print(f"{config['M']:>4} {config['ef_construction']:>6} {config['ef_search']:>6} {'ERROR':>8}")

    return results


def main():
    """Main benchmark entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="HNSW Benchmark")
    parser.add_argument("--n-vectors", type=int, default=2000, help="Number of vectors")
    parser.add_argument("--dimension", type=int, default=384, help="Vector dimension")
    parser.add_argument("--n-queries", type=int, default=100, help="Number of queries")
    parser.add_argument("--k", type=int, default=10, help="Top-k for recall")
    parser.add_argument("--M", type=int, default=16, help="HNSW M parameter")
    parser.add_argument("--ef-construction", type=int, default=200, help="ef_construction")
    parser.add_argument("--ef-search", type=int, default=50, help="ef_search")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if args.sweep:
        run_parameter_sweep()
    else:
        run_benchmark(
            n_vectors=args.n_vectors,
            dimension=args.dimension,
            n_queries=args.n_queries,
            k=args.k,
            seed=args.seed,
            M=args.M,
            ef_construction=args.ef_construction,
            ef_search=args.ef_search,
        )


if __name__ == "__main__":
    main()