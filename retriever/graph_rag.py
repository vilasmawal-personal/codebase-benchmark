from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict


class GraphRAGRetriever:
    """
    Graph-based retriever using:
    - Node embeddings (semantic search)
    - Graph expansion (multi-hop reasoning)

    Input:
        nodes: list of node dicts (with embeddings)
        edges: list of edge dicts

    Output:
        indices of relevant nodes
    """

    def __init__(
        self,
        nodes: List[Dict],
        edges: List[Dict],
        alpha: float = 0.7,
        expand_hops: int = 1,
    ):
        """
        Args:
            nodes: graph nodes with embeddings
            edges: graph edges
            alpha: weight for embedding vs graph score
            expand_hops: how many hops to expand
        """

        self.nodes = nodes
        self.edges = edges
        self.alpha = alpha
        self.expand_hops = expand_hops

        # Build embedding matrix
        self.embeddings = np.array([n["embedding"] for n in nodes])

        # Build adjacency list
        self.graph = self._build_graph(edges)

    # --------------------------------------------------
    # 🔹 Build adjacency
    # --------------------------------------------------
    def _build_graph(self, edges: List[Dict]):
        graph = defaultdict(list)

        for e in edges:
            src = e["source"]
            dst = e["target"]

            graph[src].append(dst)
            graph[dst].append(src)  # undirected for now

        return graph

    # --------------------------------------------------
    # 🔹 Main search
    # --------------------------------------------------
    def search(
        self,
        query: str,
        embedder,
        top_k: int = 25,
    ) -> List[int]:
        """
        Retrieve top-k node indices
        """

        # Step 1: Embed query
        q_emb = embedder.embed_query(query)

        # Step 2: Semantic similarity
        sim_scores = self._cosine_similarity(q_emb)

        # Step 3: Graph expansion
        graph_scores = self._graph_expansion(sim_scores)

        # Step 4: Combine scores
        final_scores = (
            self.alpha * sim_scores +
            (1 - self.alpha) * graph_scores
        )

        # Step 5: Rank
        top_indices = np.argsort(final_scores)[::-1][:top_k]

        return top_indices.tolist()

    # --------------------------------------------------
    # 🔹 Cosine similarity
    # --------------------------------------------------
    def _cosine_similarity(self, query_emb: np.ndarray) -> np.ndarray:
        """
        Compute similarity between query and all nodes
        """

        return self.embeddings @ query_emb

    # --------------------------------------------------
    # 🔹 Graph expansion
    # --------------------------------------------------
    def _graph_expansion(self, base_scores: np.ndarray) -> np.ndarray:
        """
        Expand scores via graph neighbors (multi-hop)
        """

        expanded_scores = np.zeros_like(base_scores)

        for node_idx, score in enumerate(base_scores):
            if score <= 0:
                continue

            # BFS-like expansion
            visited = set()
            queue = [(node_idx, 0)]

            while queue:
                current, depth = queue.pop(0)

                if current in visited or depth > self.expand_hops:
                    continue

                visited.add(current)

                # decay score with depth
                decay = 1 / (depth + 1)

                expanded_scores[current] += score * decay

                # expand neighbors
                for neighbor in self.graph.get(current, []):
                    queue.append((neighbor, depth + 1))

        return expanded_scores

    # --------------------------------------------------
    # 🔹 Retrieve nodes (helper)
    # --------------------------------------------------
    def get_nodes(self, indices: List[int]) -> List[Dict]:
        return [self.nodes[i] for i in indices]

    # --------------------------------------------------
    # 🔹 Debug: return scores
    # --------------------------------------------------
    def search_with_scores(
        self,
        query: str,
        embedder,
        top_k: int = 25,
    ) -> List[Tuple[int, float]]:
        """
        Returns (index, score)
        """

        q_emb = embedder.embed_query(query)

        sim_scores = self._cosine_similarity(q_emb)
        graph_scores = self._graph_expansion(sim_scores)

        final_scores = (
            self.alpha * sim_scores +
            (1 - self.alpha) * graph_scores
        )

        top_indices = np.argsort(final_scores)[::-1][:top_k]

        return [(int(i), float(final_scores[i])) for i in top_indices]

    def __repr__(self):
        return f"GraphRAGRetriever(nodes={len(self.nodes)}, alpha={self.alpha}, hops={self.expand_hops})"