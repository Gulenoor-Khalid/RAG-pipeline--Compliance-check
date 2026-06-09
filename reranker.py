"""
reranker.py — Production RAG reranker with heuristic and CrossEncoder options
Supports multiple reranking strategies for construction compliance analysis
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any
import warnings

from ingestion import Chunk


class RerankMode(Enum):
    HEURISTIC = "heuristic"
    CROSS_ENCODER = "cross_encoder"
    COMPARE_BOTH = "compare_both"


@dataclass
class HeuristicWeights:
    """Configurable weights for heuristic reranking"""
    retrieval_score: float = 0.6
    compliance_signal: float = 0.15
    numeric_density: float = 0.15
    query_overlap: float = 0.1
    
    def normalize(self):
        """Ensure weights sum to 1.0"""
        total = self.retrieval_score + self.compliance_signal + self.numeric_density + self.query_overlap
        if total != 1.0:
            self.retrieval_score /= total
            self.compliance_signal /= total
            self.numeric_density /= total
            self.query_overlap /= total


@dataclass
class RerankScore:
    """Detailed scoring breakdown for explainability"""
    chunk_id: str
    original_retrieval_score: float
    compliance_signal: float
    numeric_density: float
    query_overlap: float
    heuristic_score: float
    cross_encoder_score: Optional[float] = None
    final_score: float = 0.0
    rank_before: int = 0
    rank_after: int = 0


class Reranker:
    """
    Production-grade reranker supporting multiple strategies:
    - Configurable heuristic reranking
    - CrossEncoder neural reranking 
    - Comparative analysis mode
    """

    def __init__(
        self, 
        mode: RerankMode = RerankMode.HEURISTIC,
        heuristic_weights: Optional[HeuristicWeights] = None,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.mode = mode
        self.heuristic_weights = heuristic_weights or HeuristicWeights()
        self.heuristic_weights.normalize()
        self.cross_encoder_model = cross_encoder_model
        self._cross_encoder = None
        
        # Construction compliance keywords (higher weight for these)
        self.compliance_keywords = {
            "minimum", "maximum", "shall", "must", "required", "compliance", "standard",
            "code", "regulation", "specification", "fire", "safety", "structural",
            "concrete", "steel", "width", "height", "thickness", "pressure", "load",
            "exit", "evacuation", "sprinkler", "ventilation", "electrical", "plumbing",
            "diameter", "spacing", "clearance", "tolerance", "capacity", "resistance"
        }
    
    def _load_cross_encoder(self):
        """Lazy load CrossEncoder model"""
        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
                self._cross_encoder = CrossEncoder(self.cross_encoder_model)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for CrossEncoder reranking.\n"
                    "Run: pip install sentence-transformers"
                )
            except Exception as e:
                warnings.warn(f"Failed to load CrossEncoder model: {e}")
                self._cross_encoder = "failed"
        return self._cross_encoder if self._cross_encoder != "failed" else None

    
    def _compliance_signal(self, chunk: Chunk, query: str) -> float:
        """Score based on compliance keywords in text and query"""
        text = chunk.text.lower()
        query_lower = query.lower()
        
        # Count compliance keywords in chunk
        keyword_count = sum(1 for kw in self.compliance_keywords if kw in text)
        keyword_density = keyword_count / max(len(text.split()), 1)
        
        # Boost if query contains compliance terms
        query_compliance = sum(1 for kw in self.compliance_keywords if kw in query_lower)
        query_boost = 1.0 + (query_compliance * 0.1)
        
        # Additional boost for document type relevance
        type_boost = 1.0
        if "specification" in query_lower or "codebook" in query_lower:
            if chunk.document_type == "project":
                type_boost = 1.2
        elif "code" in query_lower or "standard" in query_lower:
            if chunk.document_type == "code":
                type_boost = 1.2
        
        return min(keyword_density * query_boost * type_boost, 1.0)

    def _numeric_density(self, chunk: Chunk) -> float:
        """Score based on numeric content (specifications, measurements)"""
        text = chunk.text
        
        # Find numbers with units (e.g., "300mm", "2.5m", "25%", "1200°C")
        numeric_patterns = [
            r'\d+\.?\d*\s*(?:mm|cm|m|km|ft|in|%|°C|°F|kN|MPa|psi|kg|lb)',
            r'\d+\.?\d*\s*(?:minutes?|hours?|seconds?|days?)',
            r'\d+\.?\d*\s*(?:litres?|gallons?)',
            r'\b\d+\.?\d*\b'  # Any number
        ]
        
        total_numbers = 0
        for pattern in numeric_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            total_numbers += len(matches)
        
        # Normalize by text length
        words = len(text.split())
        return min(total_numbers / max(words, 1), 1.0)

    def _query_overlap(self, chunk: Chunk, query: str) -> float:
        """Score based on exact term overlap with query"""
        chunk_tokens = set(re.findall(r'\w+', chunk.text.lower()))
        query_tokens = set(re.findall(r'\w+', query.lower()))
        
        if not query_tokens:
            return 0.0
        
        overlap = len(chunk_tokens.intersection(query_tokens))
        return overlap / len(query_tokens)
    
    def _calculate_heuristic_scores(
        self, 
        query: str, 
        retrieved_results: List[Tuple[Chunk, float]]
    ) -> List[RerankScore]:
        """Calculate heuristic scores for all chunks"""
        scores = []
        
        for rank, (chunk, original_score) in enumerate(retrieved_results, 1):
            # Calculate individual signals
            compliance = self._compliance_signal(chunk, query)
            numeric = self._numeric_density(chunk)
            overlap = self._query_overlap(chunk, query)
            
            # Combine signals with configurable weights
            heuristic_score = (
                original_score * self.heuristic_weights.retrieval_score +
                compliance * self.heuristic_weights.compliance_signal +
                numeric * self.heuristic_weights.numeric_density +
                overlap * self.heuristic_weights.query_overlap
            )
            
            score_obj = RerankScore(
                chunk_id=chunk.chunk_id,
                original_retrieval_score=original_score,
                compliance_signal=compliance,
                numeric_density=numeric,
                query_overlap=overlap,
                heuristic_score=heuristic_score,
                rank_before=rank
            )
            scores.append(score_obj)
        
        return scores
    
    def _calculate_cross_encoder_scores(
        self, 
        query: str, 
        retrieved_results: List[Tuple[Chunk, float]]
    ) -> Dict[str, float]:
        """Calculate CrossEncoder scores for query-chunk pairs"""
        cross_encoder = self._load_cross_encoder()
        
        if cross_encoder is None:
            # Return zeros if CrossEncoder unavailable
            return {chunk.chunk_id: 0.0 for chunk, _ in retrieved_results}
        
        # Prepare query-chunk pairs for CrossEncoder
        pairs = []
        chunk_ids = []
        
        for chunk, _ in retrieved_results:
            pairs.append([query, chunk.text])
            chunk_ids.append(chunk.chunk_id)
        
        # Get CrossEncoder scores
        try:
            ce_scores = cross_encoder.predict(pairs)
            return dict(zip(chunk_ids, ce_scores.tolist()))
        except Exception as e:
            warnings.warn(f"CrossEncoder prediction failed: {e}")
            return {chunk_id: 0.0 for chunk_id in chunk_ids}
    
    def rerank(
        self, 
        query: str, 
        retrieved_results: List[Tuple[Chunk, float]]
    ) -> Tuple[List[Tuple[Chunk, float]], List[RerankScore]]:
        """
        Rerank results using the configured strategy.
        
        Args:
            query: Search query
            retrieved_results: List of (Chunk, retrieval_score) from initial retrieval
        
        Returns:
            - Reranked list of (Chunk, final_score)
            - List of RerankScore objects with detailed scoring breakdown
        """
        if not retrieved_results:
            return [], []
        
        # Calculate heuristic scores
        scores = self._calculate_heuristic_scores(query, retrieved_results)
        
        # Calculate CrossEncoder scores if needed
        cross_encoder_scores = {}
        if self.mode in [RerankMode.CROSS_ENCODER, RerankMode.COMPARE_BOTH]:
            cross_encoder_scores = self._calculate_cross_encoder_scores(query, retrieved_results)
        
        # Add CrossEncoder scores to score objects
        for score in scores:
            score.cross_encoder_score = cross_encoder_scores.get(score.chunk_id)
        
        # Determine final scores based on mode
        if self.mode == RerankMode.HEURISTIC:
            for score in scores:
                score.final_score = score.heuristic_score
        
        elif self.mode == RerankMode.CROSS_ENCODER:
            for score in scores:
                score.final_score = score.cross_encoder_score or score.original_retrieval_score
        
        elif self.mode == RerankMode.COMPARE_BOTH:
            # Combine heuristic and CrossEncoder with equal weight
            for score in scores:
                ce_score = score.cross_encoder_score or 0.0
                score.final_score = (score.heuristic_score + ce_score) / 2.0
        
        # Sort by final score and assign new ranks
        scores.sort(key=lambda x: x.final_score, reverse=True)
        for new_rank, score in enumerate(scores, 1):
            score.rank_after = new_rank
        
        # Create reranked results
        chunk_map = {c.chunk_id: c for c, _ in retrieved_results}
        reranked = [
            (chunk_map[score.chunk_id], score.final_score) 
            for score in scores
        ]
        
        return reranked, scores
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get current configuration for debugging/logging"""
        return {
            "mode": self.mode.value,
            "heuristic_weights": {
                "retrieval_score": self.heuristic_weights.retrieval_score,
                "compliance_signal": self.heuristic_weights.compliance_signal,
                "numeric_density": self.heuristic_weights.numeric_density,
                "query_overlap": self.heuristic_weights.query_overlap,
            },
            "cross_encoder_model": self.cross_encoder_model,
            "cross_encoder_loaded": self._cross_encoder is not None and self._cross_encoder != "failed"
        }