"""
evaluator.py — Evaluation metrics for RAG system performance
"""

import re
from dataclasses import dataclass
from typing import List, Optional

from ingestion import Chunk


@dataclass
class GoldenExample:
    """Ground truth example for evaluation"""
    question: str
    expected_answer: str
    discipline: str


@dataclass
class EvalResult:
    """Complete evaluation result"""
    retrieval_score: float  # 0-100, overall retrieval effectiveness
    hallucination_risk: float  # 0-100, risk of hallucinated content
    is_correct: Optional[bool]  # True/False/None if golden dataset match found
    correctness_note: str  # Explanation of correctness assessment
    matched_golden: Optional[GoldenExample]  # The golden example that matched, if any
    
    # New detailed metrics
    retrieval_hit_rate: float  # 0-100, did expected chunk appear in top-K?
    citation_coverage: float  # 0-100, percentage of answer claims supported by chunks
    context_utilization: float  # 0-100, how many chunks contributed to answer
    retrieval_precision: float  # 0-100, percentage of retrieved chunks relevant to query
    faithfulness_score: float  # 0-100, answer grounded in retrieved context


class Evaluator:
    """Evaluates RAG system performance using multiple metrics"""
    
    def __init__(self):
        # Small golden dataset for demonstration
        # In production, this would be loaded from a file or database
        self.golden_dataset = [
            GoldenExample(
                question="What is the minimum fire exit width?",
                expected_answer="Minimum exit width is typically 32 inches (813mm) for doors and 44 inches (1118mm) for corridors",
                discipline="fire_safety"
            ),
            GoldenExample(
                question="What is the minimum concrete cover for reinforcement?",
                expected_answer="Concrete cover varies by exposure: 20mm minimum for indoor, 40mm for outdoor exposure",
                discipline="structural"
            ),
            # Add more golden examples as needed
        ]
    
    def _calculate_retrieval_hit_rate(self, query: str, chunks: List[Chunk]) -> float:
        """
        Calculate hit rate - did the expected chunk appear in top-K results?
        For demo purposes, we estimate this by checking if highly relevant chunks are present.
        In production, this would use labeled ground truth chunks.
        """
        if not chunks:
            return 0.0
            
        query_tokens = set(re.findall(r'\w+', query.lower()))
        
        # Find the most relevant chunk (highest overlap)
        best_overlap = 0.0
        for chunk in chunks:
            chunk_tokens = set(re.findall(r'\w+', chunk.text.lower()))
            if query_tokens:
                overlap = len(chunk_tokens.intersection(query_tokens)) / len(query_tokens)
                best_overlap = max(best_overlap, overlap)
        
        # Consider hit if best chunk has >70% overlap
        return 100.0 if best_overlap > 0.7 else best_overlap * 100.0
    
    def _calculate_citation_coverage(self, answer: str, chunks: List[Chunk]) -> float:
        """
        Calculate what percentage of answer claims are supported by retrieved chunks.
        """
        if not answer or not chunks:
            return 0.0
        
        # Extract factual claims from answer (sentences with specific info)
        answer_sentences = [s.strip() for s in re.split(r'[.!?]', answer) if s.strip()]
        factual_claims = []
        
        for sentence in answer_sentences:
            # Consider a sentence a "claim" if it contains:
            # - Numbers/measurements
            # - Specific technical terms
            # - Definitive statements (must, shall, required, minimum, maximum)
            if (re.search(r'\d+', sentence) or 
                re.search(r'\b(minimum|maximum|required|must|shall|standard|code)\b', sentence.lower()) or
                len(sentence.split()) > 5):  # Substantive sentences
                factual_claims.append(sentence)
        
        if not factual_claims:
            return 85.0  # High coverage if no specific claims to verify
        
        # Check how many claims are supported by chunks
        chunk_text = " ".join(c.text for c in chunks).lower()
        supported_claims = 0
        
        for claim in factual_claims:
            claim_terms = re.findall(r'\w+', claim.lower())
            # Remove common words
            significant_terms = [t for t in claim_terms if len(t) > 2 and 
                               t not in {'the', 'and', 'for', 'are', 'this', 'that', 'with', 'have', 'will'}]
            
            if significant_terms:
                found_terms = sum(1 for term in significant_terms if term in chunk_text)
                support_ratio = found_terms / len(significant_terms)
                if support_ratio >= 0.6:  # 60% of terms found
                    supported_claims += 1
        
        return (supported_claims / len(factual_claims)) * 100.0
    
    def _calculate_context_utilization(self, answer: str, chunks: List[Chunk]) -> float:
        """
        Calculate how many retrieved chunks actually contributed to the answer.
        """
        if not answer or not chunks:
            return 0.0
        
        answer_lower = answer.lower()
        utilized_chunks = 0
        
        for chunk in chunks:
            # Check if chunk content appears in answer
            chunk_terms = re.findall(r'\w{4,}', chunk.text.lower())  # Words 4+ chars
            significant_terms = [t for t in chunk_terms if 
                               t not in {'this', 'that', 'with', 'have', 'will', 'should', 'would'}]
            
            if significant_terms:
                found_in_answer = sum(1 for term in significant_terms[:10] if term in answer_lower)  # Check first 10 terms
                if found_in_answer >= 2:  # At least 2 terms from chunk appear in answer
                    utilized_chunks += 1
        
        return (utilized_chunks / len(chunks)) * 100.0
    
    def _calculate_retrieval_precision(self, query: str, chunks: List[Chunk]) -> float:
        """
        Calculate what percentage of retrieved chunks are relevant to the query.
        """
        if not chunks:
            return 0.0
        
        query_tokens = set(re.findall(r'\w+', query.lower()))
        if not query_tokens:
            return 0.0
        
        relevant_chunks = 0
        
        for chunk in chunks:
            chunk_tokens = set(re.findall(r'\w+', chunk.text.lower()))
            overlap = len(chunk_tokens.intersection(query_tokens)) / len(query_tokens)
            
            # Also check for semantic relevance indicators
            semantic_relevance = 0
            if any(discipline in query.lower() for discipline in 
                   ['fire', 'structural', 'electrical', 'plumbing', 'ventilation']):
                if chunk.discipline in query.lower():
                    semantic_relevance += 0.2
            
            # Consider relevant if >40% overlap or strong semantic match
            if overlap > 0.4 or semantic_relevance > 0.15:
                relevant_chunks += 1
        
        return (relevant_chunks / len(chunks)) * 100.0
    
    def _calculate_faithfulness_score(self, answer: str, chunks: List[Chunk]) -> float:
        """
        Enhanced faithfulness score - estimate whether answer content is grounded in retrieved context.
        This is similar to hallucination risk but focuses on positive grounding evidence.
        """
        if not answer or not chunks:
            return 0.0
        
        # Extract all substantive statements from answer
        answer_sentences = [s.strip() for s in re.split(r'[.!?]', answer) if s.strip()]
        
        total_grounding_score = 0.0
        sentence_count = 0
        
        chunk_text = " ".join(c.text for c in chunks).lower()
        
        for sentence in answer_sentences:
            if len(sentence.split()) < 4:  # Skip very short sentences
                continue
                
            sentence_count += 1
            sentence_lower = sentence.lower()
            
            # Extract key terms from sentence
            sentence_terms = re.findall(r'\w{3,}', sentence_lower)  # 3+ char words
            significant_terms = [t for t in sentence_terms if 
                               t not in {'the', 'and', 'for', 'are', 'this', 'that', 'with', 'have', 'will', 'can', 'may'}]
            
            if significant_terms:
                # Check grounding strength
                found_terms = sum(1 for term in significant_terms if term in chunk_text)
                grounding_ratio = found_terms / len(significant_terms)
                
                # Bonus for exact phrase matches
                phrase_bonus = 0
                words = sentence.split()
                for i in range(len(words) - 2):
                    phrase = " ".join(words[i:i+3]).lower()
                    if phrase in chunk_text:
                        phrase_bonus += 0.1
                
                sentence_grounding = min(grounding_ratio + phrase_bonus, 1.0)
                total_grounding_score += sentence_grounding
        
        if sentence_count == 0:
            return 50.0  # Neutral score for empty/very short answers
        
        avg_grounding = total_grounding_score / sentence_count
        return avg_grounding * 100.0
    
    def _calculate_retrieval_score(self, query: str, chunks: List[Chunk]) -> float:
        """
        Enhanced retrieval effectiveness score combining multiple factors.
        """
        if not chunks:
            return 0.0
        
        # Component scores
        hit_rate = self._calculate_retrieval_hit_rate(query, chunks)
        precision = self._calculate_retrieval_precision(query, chunks)
        
        # Diversity bonus
        unique_disciplines = len(set(c.discipline for c in chunks))
        unique_sources = len(set(c.source_doc for c in chunks))
        diversity_score = min((unique_disciplines + unique_sources) * 10, 30)
        
        # Technical content bonus
        technical_chunks = sum(1 for c in chunks if 
                             re.search(r'\d+\.?\d*\s*(?:mm|cm|m|ft|in|%|°C|°F|kN|MPa|psi)', c.text))
        technical_bonus = min(technical_chunks * 15, 25)
        
        # Combine scores with weights
        base_score = (hit_rate * 0.4 + precision * 0.4) 
        total_score = base_score + diversity_score + technical_bonus
        
        return min(total_score, 100.0)
    
    def _assess_hallucination_risk(self, answer: str, chunks: List[Chunk]) -> float:
        """
        Estimate hallucination risk by checking if answer content 
        is grounded in the provided chunks
        """
        if not answer or not chunks:
            return 100.0
        
        # Extract key claims from answer (sentences with numbers/specifications)
        answer_sentences = [s.strip() for s in re.split(r'[.!?]', answer) if s.strip()]
        technical_claims = [s for s in answer_sentences if re.search(r'\d+', s)]
        
        if not technical_claims:
            return 25.0  # Low risk if no specific technical claims
        
        # Check how many technical claims can be found in chunks
        grounded_claims = 0
        chunk_text = " ".join(c.text for c in chunks).lower()
        
        for claim in technical_claims:
            # Extract key terms from claim
            claim_terms = re.findall(r'\w+', claim.lower())
            # Check if most terms appear in chunk text
            found_terms = sum(1 for term in claim_terms if term in chunk_text)
            if found_terms >= len(claim_terms) * 0.6:  # 60% threshold
                grounded_claims += 1
        
        if not technical_claims:
            return 30.0
        
        grounding_ratio = grounded_claims / len(technical_claims)
        risk = (1.0 - grounding_ratio) * 100
        
        return max(risk, 10.0)  # Minimum 10% risk
    
    def _check_correctness(self, query: str, answer: str) -> tuple[Optional[bool], str, Optional[GoldenExample]]:
        """
        Check answer correctness against golden dataset.
        Returns (is_correct, explanation, matched_golden_example)
        """
        query_lower = query.lower()
        
        # Find matching golden example
        best_match = None
        best_score = 0.0
        
        for golden in self.golden_dataset:
            # Simple keyword matching for demo
            golden_tokens = set(re.findall(r'\w+', golden.question.lower()))
            query_tokens = set(re.findall(r'\w+', query_lower))
            
            if golden_tokens and query_tokens:
                overlap = len(golden_tokens.intersection(query_tokens)) / len(golden_tokens)
                if overlap > best_score and overlap > 0.5:  # 50% threshold
                    best_score = overlap
                    best_match = golden
        
        if not best_match:
            return None, "No matching golden example found for this query", None
        
        # Simple correctness check - look for key terms from expected answer
        expected_terms = re.findall(r'\d+\s*(?:mm|cm|m|ft|in|inches|%)', best_match.expected_answer.lower())
        answer_terms = re.findall(r'\d+\s*(?:mm|cm|m|ft|in|inches|%)', answer.lower())
        
        if expected_terms:
            # Check if any expected measurements are mentioned
            found_expected = any(term in answer.lower() for term in expected_terms)
            if found_expected:
                return True, f"Answer contains expected technical specifications", best_match
            else:
                return False, f"Answer missing expected measurements: {', '.join(expected_terms)}", best_match
        else:
            # For non-technical questions, do basic keyword matching
            expected_keywords = re.findall(r'\w+', best_match.expected_answer.lower())
            answer_keywords = re.findall(r'\w+', answer.lower())
            
            common = set(expected_keywords).intersection(set(answer_keywords))
            if len(common) >= len(expected_keywords) * 0.4:  # 40% keyword overlap
                return True, f"Answer contains sufficient relevant keywords", best_match
            else:
                return False, f"Answer lacks key concepts from expected response", best_match


def evaluate(query: str, answer: str, context_chunks: List[Chunk]) -> EvalResult:
    """
    Main evaluation function - assesses RAG system performance with comprehensive metrics.
    
    Args:
        query: User's original question
        answer: LLM's generated answer  
        context_chunks: Retrieved chunks used for context
        
    Returns:
        EvalResult with all evaluation metrics
    """
    evaluator = Evaluator()
    
    # Calculate all metrics
    retrieval_score = evaluator._calculate_retrieval_score(query, context_chunks)
    hallucination_risk = evaluator._assess_hallucination_risk(answer, context_chunks)
    is_correct, correctness_note, matched_golden = evaluator._check_correctness(query, answer)
    
    # New detailed metrics
    retrieval_hit_rate = evaluator._calculate_retrieval_hit_rate(query, context_chunks)
    citation_coverage = evaluator._calculate_citation_coverage(answer, context_chunks)
    context_utilization = evaluator._calculate_context_utilization(answer, context_chunks)
    retrieval_precision = evaluator._calculate_retrieval_precision(query, context_chunks)
    faithfulness_score = evaluator._calculate_faithfulness_score(answer, context_chunks)
    
    return EvalResult(
        retrieval_score=retrieval_score,
        hallucination_risk=hallucination_risk,
        is_correct=is_correct,
        correctness_note=correctness_note,
        matched_golden=matched_golden,
        retrieval_hit_rate=retrieval_hit_rate,
        citation_coverage=citation_coverage,
        context_utilization=context_utilization,
        retrieval_precision=retrieval_precision,
        faithfulness_score=faithfulness_score
    )