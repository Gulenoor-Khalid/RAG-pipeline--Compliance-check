"""
retrieval.py — FAISS vector store + BM25 keyword search + hybrid fusion
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ingestion import Chunk, embed_chunks, _get_embedder

# ---------------------------------------------------------------------------
# FAISS index wrapper
# ---------------------------------------------------------------------------

class VectorStore:
    """
    Thin wrapper around FAISS (with a pure-numpy fallback for environments
    where faiss is not installed).
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        self._chunks: List[Chunk] = []
        self._matrix: Optional[np.ndarray] = None  # (N, dim) float32
        self._faiss_index = None

    # ------------------------------------------------------------------
    def build(self, chunks: List[Chunk]) -> None:
        self._chunks = chunks
        vecs = np.stack([c.embedding for c in chunks]).astype(np.float32)
        # L2-normalise so dot product == cosine similarity
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
        self._matrix = vecs / norms

        try:
            import faiss
            idx = faiss.IndexFlatIP(self.dim)
            idx.add(self._matrix)
            self._faiss_index = idx
        except ImportError:
            self._faiss_index = None  # fall back to numpy

    # ------------------------------------------------------------------
    def query(
        self,
        query_vec: np.ndarray,
        top_k: int = 8,
        metadata_filter: Optional[Dict] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Returns list of (Chunk, cosine_similarity) sorted descending.
        metadata_filter: dict of {field: value} that must match chunk metadata.
        """
        q = query_vec.astype(np.float32)
        q /= np.linalg.norm(q) + 1e-8

        if self._faiss_index is not None:
            scores, idxs = self._faiss_index.search(q.reshape(1, -1), len(self._chunks))
            pairs = [(self._chunks[i], float(scores[0][j])) for j, i in enumerate(idxs[0]) if i >= 0]
        else:
            sims = (self._matrix @ q).tolist()
            pairs = [(c, float(s)) for c, s in zip(self._chunks, sims)]
            pairs.sort(key=lambda x: x[1], reverse=True)

        # Apply metadata filter
        if metadata_filter:
            pairs = [
                (c, s) for c, s in pairs
                if all(getattr(c, k, None) == v for k, v in metadata_filter.items())
            ]

        return pairs[:top_k]


# ---------------------------------------------------------------------------
# BM25 (simple in-memory implementation)
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    """
    Robertson BM25 over the chunk corpus.
    k1=1.5, b=0.75
    """
    k1 = 1.5
    b  = 0.75

    def __init__(self):
        self._chunks: List[Chunk] = []
        self._tf: List[Dict[str, int]] = []
        self._df: Dict[str, int] = defaultdict(int)
        self._avg_dl: float = 0.0
        self._N: int = 0

    def build(self, chunks: List[Chunk]) -> None:
        self._chunks = chunks
        self._N = len(chunks)
        total_len = 0
        for chunk in chunks:
            tokens = _tokenise(chunk.text)
            total_len += len(tokens)
            tf: Dict[str, int] = defaultdict(int)
            for t in tokens:
                tf[t] += 1
            self._tf.append(dict(tf))
            for t in set(tokens):
                self._df[t] += 1
        self._avg_dl = total_len / max(self._N, 1)

    def query(
        self,
        query_text: str,
        top_k: int = 8,
        metadata_filter: Optional[Dict] = None,
    ) -> List[Tuple[Chunk, float]]:
        tokens = _tokenise(query_text)
        scores = []
        for i, (chunk, tf) in enumerate(zip(self._chunks, self._tf)):
            score = 0.0
            dl = sum(tf.values())
            for t in tokens:
                if t not in tf:
                    continue
                n_t = self._df.get(t, 0)
                idf = math.log((self._N - n_t + 0.5) / (n_t + 0.5) + 1)
                tf_t = tf[t]
                score += idf * (tf_t * (self.k1 + 1)) / (
                    tf_t + self.k1 * (1 - self.b + self.b * dl / self._avg_dl)
                )
            scores.append((chunk, score))

        if metadata_filter:
            scores = [
                (c, s) for c, s in scores
                if all(getattr(c, k, None) == v for k, v in metadata_filter.items())
            ]

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ---------------------------------------------------------------------------
# Hybrid retriever — reciprocal rank fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[Chunk, float]]],
    k: int = 60,
) -> List[Tuple[Chunk, float]]:
    """
    RRF merges multiple ranked lists.  Score = Σ 1/(k + rank_i).
    """
    scores: Dict[str, float] = defaultdict(float)
    chunk_map: Dict[str, Chunk] = {}

    for ranked in ranked_lists:
        for rank, (chunk, _) in enumerate(ranked, start=1):
            cid = chunk.chunk_id
            scores[cid] += 1.0 / (k + rank)
            chunk_map[cid] = chunk

    merged = [(chunk_map[cid], sc) for cid, sc in scores.items()]
    merged.sort(key=lambda x: x[1], reverse=True)
    return merged


# ---------------------------------------------------------------------------
# Public Retriever class
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Combines FAISS vector search + BM25 with RRF fusion and metadata filtering.
    """

    def __init__(self):
        self.vector_store = VectorStore()
        self.bm25 = BM25Index()
        self._built = False

    def build(self, chunks: List[Chunk]) -> None:
        self.vector_store.build(chunks)
        self.bm25.build(chunks)
        self._built = True

    def _embed_query(self, query: str) -> np.ndarray:
        embedder = _get_embedder()
        vec = embedder.encode([query], normalize_embeddings=True)[0]
        return vec.astype(np.float32)

    def retrieve(
        self,
        query: str,
        top_k: int = 6,
        metadata_filter: Optional[Dict] = None,
        use_hybrid: bool = True,
    ) -> List[Tuple[Chunk, float]]:
        """
        Returns list of (Chunk, rrf_score) after hybrid fusion.
        """
        q_vec = self._embed_query(query)

        vector_results = self.vector_store.query(q_vec, top_k=top_k * 2, metadata_filter=metadata_filter)

        if use_hybrid:
            bm25_results = self.bm25.query(query, top_k=top_k * 2, metadata_filter=metadata_filter)
            merged = reciprocal_rank_fusion([vector_results, bm25_results])
        else:
            merged = vector_results

        return merged[:top_k]
