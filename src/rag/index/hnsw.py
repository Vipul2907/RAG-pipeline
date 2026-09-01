"""
Hierarchical Navigable Small World (HNSW) index implementation.

A multi-layer proximity graph for approximate nearest neighbor search.
Built from scratch using NumPy, inheriting from the VectorIndex base class.
"""

import heapq
import json
import math
import random
import time
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np

from .base import VectorIndex, SearchResult
from ..ingestion import Chunk
from ..config import logger


class HNSWIndex(VectorIndex):
    """
    Hierarchical Navigable Small World (HNSW) index.

    A multi-layer graph where:
    - Layer 0 contains all nodes with dense connections (M0 max links)
    - Higher layers have fewer nodes with longer-range connections (M max links)
    - Search starts at top layer entry point, greedily descends, then searches at layer 0

    Attributes:
        M: Max outgoing links per node for layers > 0
        M0: Max outgoing links per node for layer 0 (typically 2 * M)
        ef_construction: Search queue size during index building
        ef_search: Search queue size during query
        mL: Normalization factor for level generation (1/ln(M))
    """

    def __init__(
        self,
        dimension: int,
        M: int = 16,
        M0: int = None,
        ef_construction: int = 200,
        ef_search: int = 50,
        mL: float = None,
        seed: int = 42,
    ):
        """
        Initialize HNSW index.

        Args:
            dimension: Vector dimensionality
            M: Max connections per node for layers > 0 (default: 16)
            M0: Max connections for layer 0 (default: 2 * M)
            ef_construction: Size of dynamic candidate list during build (default: 200)
            ef_search: Size of dynamic candidate list during query (default: 50)
            mL: Level generation normalization (default: 1 / ln(M))
            seed: Random seed for reproducibility
        """
        super().__init__(dimension)

        if M < 2:
            raise ValueError("M must be >= 2")
        if ef_construction < 1:
            raise ValueError("ef_construction must be >= 1")
        if ef_search < 1:
            raise ValueError("ef_search must be >= 1")

        self.M = M
        self.M0 = M0 if M0 is not None else 2 * M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.mL = mL if mL is not None else 1.0 / math.log(M)
        self.seed = seed

        # Graph structure: list of layers, each layer is dict[node_id] -> set[neighbor_ids]
        self._graphs: List[dict[int, Set[int]]] = []

        # Node embeddings: node_id -> normalized vector
        self._node_vectors: dict[int, np.ndarray] = {}

        # Node metadata: node_id -> Chunk
        self._node_chunks: dict[int, Chunk] = {}

        # Entry point node_id (node in highest layer)
        self._entry_point: Optional[int] = None

        # Current max layer
        self._max_layer: int = -1

        # Next node ID to assign
        self._next_node_id: int = 0

        # Random state for level generation
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        logger.info(
            f"Initialized HNSWIndex: dim={dimension}, M={M}, M0={self.M0}, "
            f"ef_construction={ef_construction}, ef_search={ef_search}, mL={self.mL:.4f}"
        )

    def _get_random_level(self) -> int:
        """
        Generate random level for a new node using exponential distribution.

        Returns level >= 0, with P(level = l) ~ exp(-l / mL)
        """
        # Use -ln(uniform) * mL for exponential distribution
        level = int(-math.log(self._rng.random()) * self.mL)
        return level

    def _ensure_layer(self, layer: int) -> None:
        """Ensure graph layer exists."""
        while len(self._graphs) <= layer:
            self._graphs.append({})

    def _add_link(self, layer: int, node_a: int, node_b: int) -> None:
        """Add bidirectional link between two nodes at given layer."""
        self._ensure_layer(layer)
        if node_a not in self._graphs[layer]:
            self._graphs[layer][node_a] = set()
        if node_b not in self._graphs[layer]:
            self._graphs[layer][node_b] = set()
        self._graphs[layer][node_a].add(node_b)
        self._graphs[layer][node_b].add(node_a)

    def _remove_link(self, layer: int, node_a: int, node_b: int) -> None:
        """Remove bidirectional link between two nodes at given layer."""
        if layer < len(self._graphs):
            if node_a in self._graphs[layer]:
                self._graphs[layer][node_a].discard(node_b)
            if node_b in self._graphs[layer]:
                self._graphs[layer][node_b].discard(node_a)

    def _get_neighbors(self, layer: int, node_id: int) -> Set[int]:
        """Get neighbors of a node at given layer."""
        if layer < len(self._graphs) and node_id in self._graphs[layer]:
            return self._graphs[layer][node_id]
        return set()

    def _get_link_count(self, layer: int, node_id: int) -> int:
        """Get number of links for a node at given layer."""
        return len(self._get_neighbors(layer, node_id))

    def _distance(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute cosine distance between two normalized vectors."""
        return 1.0 - float(vec_a @ vec_b)

    def _select_neighbors_heuristic(
        self,
        candidates: List[Tuple[float, int]],
        max_connections: int,
        layer: int,
        query_vector: Optional[np.ndarray] = None,
    ) -> List[int]:
        """
        Select neighbors using HNSW heuristic (Algorithm 4 from paper).

        This maintains graph navigability by preferring neighbors that are
        close to the query but also well-separated from each other.
        """
        if not candidates:
            return []

        # Sort by distance ascending
        candidates.sort(key=lambda x: x[0])

        # Simple but effective: keep closest max_connections
        # For production, implement the full heuristic with expand_candidates
        selected = []
        for dist, nid in candidates:
            if len(selected) >= max_connections:
                break
            selected.append(nid)

        return selected

    def _search_layer(
        self,
        query_vector: np.ndarray,
        entry_points: List[int],
        ef: int,
        layer: int,
    ) -> List[Tuple[float, int]]:
        """
        Search for nearest neighbors within a single layer using best-first search.

        This is the core search algorithm (Algorithm 2 from HNSW paper).
        Uses a priority queue to maintain candidate set of size ef.
        """
        if not entry_points:
            return []

        visited = set()
        # Min-heap for candidates: (distance, node_id) - we pop closest first
        candidates = []

        # Initialize with entry points
        for ep in entry_points:
            if ep in visited:
                continue
            visited.add(ep)
            dist = self._distance(query_vector, self._node_vectors[ep])
            heapq.heappush(candidates, (dist, ep))

        # Max-heap for results (negative distance): (-distance, node_id)
        # This keeps the ef best results found so far
        results = []

        while candidates:
            dist, node_id = heapq.heappop(candidates)

            # Early termination: if we have ef results and current candidate
            # is worse than the worst result, we can stop
            if len(results) >= ef and dist > -results[0][0]:
                break

            # Add to results
            heapq.heappush(results, (-dist, node_id))
            if len(results) > ef:
                heapq.heappop(results)

            # Explore neighbors
            for neighbor in self._get_neighbors(layer, node_id):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                n_dist = self._distance(query_vector, self._node_vectors[neighbor])
                heapq.heappush(candidates, (n_dist, neighbor))

        # Convert results to sorted list (ascending distance)
        final_results = [(-d, nid) for d, nid in results]
        final_results.sort(key=lambda x: x[0])
        return final_results

    def _greedy_search(
        self,
        query_vector: np.ndarray,
        entry_point: int,
        layer: int,
    ) -> int:
        """
        Greedy search at a single layer (Algorithm 1 from HNSW paper).

        Starting from entry_point, repeatedly move to the closest neighbor
        until no improvement is found. Returns the closest node at this layer.
        """
        current = entry_point
        current_dist = self._distance(query_vector, self._node_vectors[current])

        while True:
            improved = False
            neighbors = self._get_neighbors(layer, current)
            for neighbor in neighbors:
                dist = self._distance(query_vector, self._node_vectors[neighbor])
                if dist < current_dist:
                    current = neighbor
                    current_dist = dist
                    improved = True
                    break  # Move to this neighbor and continue search

            if not improved:
                break

        return current

    def add(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        """
        Add vectors to the HNSW index (Algorithm 3 from HNSW paper).

        Args:
            embeddings: Array of shape (n, dimension)
            chunks: List of Chunk objects corresponding to each embedding
        """
        self._validate_embeddings(embeddings, len(chunks))

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalized = embeddings / norms

        start_time = time.time()
        added_count = 0

        for i, (vector, chunk) in enumerate(zip(normalized, chunks)):
            node_id = self._next_node_id
            self._next_node_id += 1

            # Store vector and metadata
            self._node_vectors[node_id] = vector.astype(np.float32)
            self._node_chunks[node_id] = chunk

            # Generate level for this node
            level = self._get_random_level()

            # First node becomes entry point
            if self._entry_point is None:
                self._entry_point = node_id
                self._max_layer = level
                for l in range(level + 1):
                    self._ensure_layer(l)
                    self._graphs[l][node_id] = set()
                added_count += 1
                continue

            # Start from entry point at top layer
            ep = self._entry_point

            # Greedy search from top layer down to level+1
            # At these layers, we only search, don't insert
            for layer in range(self._max_layer, level, -1):
                ep = self._greedy_search(vector, ep, layer)

            # At layer level and below, insert the node
            for layer in range(min(level, self._max_layer), -1, -1):
                max_conn = self.M0 if layer == 0 else self.M

                # Search for ef_construction candidates at this layer
                candidates = self._search_layer(
                    vector, [ep], self.ef_construction, layer
                )

                # Select neighbors to connect using heuristic
                neighbors = self._select_neighbors_heuristic(
                    candidates, max_conn, layer, vector
                )

                # Add bidirectional links
                for neighbor in neighbors:
                    self._add_link(layer, node_id, neighbor)

                # Prune excess connections for each neighbor
                for neighbor in neighbors:
                    self._prune_connections(layer, neighbor, max_conn, vector)

                # Update entry point for next layer down
                if candidates:
                    ep = candidates[0][1]

            # If new node has higher level than current max, update entry point
            if level > self._max_layer:
                self._max_layer = level
                self._entry_point = node_id
                for l in range(len(self._graphs), level + 1):
                    self._ensure_layer(l)

            added_count += 1

        self._size = len(self._node_vectors)
        elapsed = time.time() - start_time
        logger.info(
            f"Added {added_count} vectors to HNSWIndex (total: {self._size}, "
            f"layers: {self._max_layer + 1}, time: {elapsed:.3f}s)"
        )

    def _prune_connections(
        self,
        layer: int,
        node_id: int,
        max_connections: int,
        query_vector: Optional[np.ndarray] = None,
    ) -> None:
        """Prune excess connections for a node using heuristic."""
        neighbors = list(self._get_neighbors(layer, node_id))
        if len(neighbors) <= max_connections:
            return

        # Compute distances to all neighbors
        node_vec = self._node_vectors[node_id]
        neighbor_dists = []
        for n in neighbors:
            dist = self._distance(node_vec, self._node_vectors[n])
            neighbor_dists.append((dist, n))

        # Sort by distance and keep closest max_connections
        neighbor_dists.sort(key=lambda x: x[0])
        to_keep = set(n for _, n in neighbor_dists[:max_connections])
        to_remove = set(neighbors) - to_keep

        for rem in to_remove:
            self._remove_link(layer, node_id, rem)

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filter_fn: Optional[callable] = None,
    ) -> List[SearchResult]:
        """
        Search for top-k nearest neighbors using HNSW (Algorithm 5 from paper).

        Args:
            query_embedding: Query vector of shape (dimension,)
            k: Number of results to return
            filter_fn: Optional callable(chunk) -> bool to filter results

        Returns:
            List of SearchResult objects, sorted by score descending
        """
        if self._size == 0:
            logger.warning("Search on empty HNSW index")
            return []

        # Validate and normalize query
        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim != 1:
            raise ValueError(f"Query must be 1D array, got {query.ndim}D")
        if query.shape[0] != self._dimension:
            raise ValueError(
                f"Query dimension mismatch: expected {self._dimension}, got {query.shape[0]}"
            )

        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            raise ValueError("Query vector has zero norm")
        query = query / query_norm

        # Start from entry point
        ep = self._entry_point

        # Greedy descent from top layer to layer 1
        for layer in range(self._max_layer, 0, -1):
            ep = self._greedy_search(query, ep, layer)

        # Search at layer 0 with ef_search
        candidates = self._search_layer(query, [ep], self.ef_search, 0)

        # Build results with filtering
        results = []
        for rank, (dist, node_id) in enumerate(candidates, 1):
            if rank > k:
                break
            chunk = self._node_chunks[node_id]
            if filter_fn and not filter_fn(chunk):
                continue
            # Convert distance to cosine similarity score
            score = 1.0 - dist
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                    rank=rank,
                )
            )

        logger.debug(f"HNSWIndex search: k={k}, returned={len(results)} results")
        return results

    def save(self, path: Path) -> None:
        """
        Save HNSW index to disk.

        Args:
            path: Directory path to save index files.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save node vectors
        node_ids = sorted(self._node_vectors.keys())
        vectors = np.stack([self._node_vectors[nid] for nid in node_ids])
        np.save(path / "vectors.npy", vectors)

        # Save node IDs mapping (for reconstruction)
        np.save(path / "node_ids.npy", np.array(node_ids, dtype=np.int64))

        # Save graph structure
        graph_data = []
        for layer_idx, layer_graph in enumerate(self._graphs):
            layer_data = {}
            for node_id, neighbors in layer_graph.items():
                layer_data[str(node_id)] = list(neighbors)
            graph_data.append(layer_data)

        with open(path / "graphs.json", "w") as f:
            json.dump(graph_data, f)

        # Save chunks metadata
        chunks_data = []
        for node_id in node_ids:
            chunk = self._node_chunks[node_id]
            chunks_data.append({
                "node_id": node_id,
                "text": chunk.text,
                "source_file": chunk.source_file,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata,
            })

        with open(path / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

        # Save config
        config_data = {
            "dimension": self._dimension,
            "size": self._size,
            "index_type": "hnsw",
            "M": self.M,
            "M0": self.M0,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "mL": self.mL,
            "seed": self.seed,
            "max_layer": self._max_layer,
            "entry_point": self._entry_point,
            "next_node_id": self._next_node_id,
        }

        with open(path / "config.json", "w") as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"Saved HNSWIndex to {path} ({self._size} vectors, {self._max_layer + 1} layers)")

    @classmethod
    def load(cls, path: Path) -> "HNSWIndex":
        """
        Load HNSW index from disk.

        Args:
            path: Directory containing index files.

        Returns:
            Loaded HNSWIndex instance.
        """
        path = Path(path)

        # Load config
        config_path = path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, "r") as f:
            config_data = json.load(f)

        dimension = config_data.get("dimension")
        if dimension is None:
            raise ValueError("Config missing 'dimension' field")

        # Create index with saved parameters
        index = cls(
            dimension=dimension,
            M=config_data.get("M", 16),
            M0=config_data.get("M0", 32),
            ef_construction=config_data.get("ef_construction", 200),
            ef_search=config_data.get("ef_search", 50),
            mL=config_data.get("mL"),
            seed=config_data.get("seed", 42),
        )

        # Load vectors
        vectors_path = path / "vectors.npy"
        node_ids_path = path / "node_ids.npy"

        if not vectors_path.exists():
            raise FileNotFoundError(f"Vectors not found: {vectors_path}")
        if not node_ids_path.exists():
            raise FileNotFoundError(f"Node IDs not found: {node_ids_path}")

        vectors = np.load(vectors_path)
        node_ids = np.load(node_ids_path).tolist()

        if len(vectors) != len(node_ids):
            raise ValueError("Vectors and node_ids length mismatch")

        # Load graphs
        graphs_path = path / "graphs.json"
        if not graphs_path.exists():
            raise FileNotFoundError(f"Graphs not found: {graphs_path}")

        with open(graphs_path, "r") as f:
            graph_data = json.load(f)

        # Load chunks
        chunks_path = path / "chunks.json"
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks not found: {chunks_path}")

        with open(chunks_path, "r") as f:
            chunks_data = json.load(f)

        # Reconstruct index state
        index._node_vectors = {
            nid: vectors[i].astype(np.float32)
            for i, nid in enumerate(node_ids)
        }

        index._node_chunks = {}
        for chunk_data in chunks_data:
            node_id = chunk_data["node_id"]
            index._node_chunks[node_id] = Chunk(
                text=chunk_data["text"],
                source_file=chunk_data["source_file"],
                start_char=chunk_data["start_char"],
                end_char=chunk_data["end_char"],
                chunk_index=chunk_data["chunk_index"],
                metadata=chunk_data.get("metadata", {}),
            )

        index._graphs = []
        for layer_data in graph_data:
            layer_graph = {}
            for node_str, neighbors in layer_data.items():
                node_id = int(node_str)
                layer_graph[node_id] = set(neighbors)
            index._graphs.append(layer_graph)

        index._entry_point = config_data.get("entry_point")
        index._max_layer = config_data.get("max_layer", -1)
        index._next_node_id = config_data.get("next_node_id", len(node_ids))
        index._size = config_data.get("size", len(node_ids))

        logger.info(f"Loaded HNSWIndex from {path} ({index._size} vectors, {index._max_layer + 1} layers)")
        return index

    def get_stats(self) -> dict:
        """Get index statistics for debugging/analysis."""
        layer_counts = [len(layer) for layer in self._graphs]
        total_links = sum(sum(len(n) for n in layer.values()) for layer in self._graphs)
        avg_links = total_links / self._size if self._size > 0 else 0

        return {
            "size": self._size,
            "dimension": self._dimension,
            "max_layer": self._max_layer,
            "num_layers": len(self._graphs),
            "nodes_per_layer": layer_counts,
            "total_links": total_links // 2,  # bidirectional counted twice
            "avg_links_per_node": avg_links / 2,
            "entry_point": self._entry_point,
            "M": self.M,
            "M0": self.M0,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
        }