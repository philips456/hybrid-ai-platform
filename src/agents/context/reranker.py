"""
src/agents/context/reranker.py
DocumentReranker — implements CRAG confidence scoring.

Based on:
- Yan et al., arXiv:2401.15884 (Corrective RAG)

CRAG strategy:
- confidence > 0.8: use RAG documents directly
- confidence 0.4-0.8: combine RAG + supplementary search
- confidence < 0.4: discard RAG, use alternative source
"""
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RankedDocument:
    """A document with its relevance score after reranking."""
    content: str
    relevance_score: float
    source: str
    confidence_level: str  # high / medium / low


class DocumentReranker:
    """
    Reranks retrieved documents and evaluates retrieval confidence.

    Implements CRAG (Corrective RAG) confidence evaluation:
    1. Score each document for relevance to the query
    2. Compute aggregate confidence
    3. Recommend retrieval strategy based on confidence

    Usage:
        reranker = DocumentReranker()
        ranked = reranker.rerank(query, documents, embeddings)
        strategy = reranker.get_crag_strategy(ranked)
    """

    CONFIDENCE_HIGH = 0.8
    CONFIDENCE_MEDIUM = 0.4

    def rerank(
        self,
        query: str,
        documents: list[str],
        query_embedding: Optional[list[float]] = None,
        doc_embeddings: Optional[list[list[float]]] = None,
    ) -> list[RankedDocument]:
        """
        Reranks documents by relevance to query.

        Args:
            query: The search query
            documents: Retrieved document strings
            query_embedding: Query vector (optional, for semantic scoring)
            doc_embeddings: Document vectors (optional)

        Returns:
            List of RankedDocument sorted by relevance descending
        """
        if not documents:
            return []

        ranked = []

        for i, doc in enumerate(documents):
            if query_embedding and doc_embeddings and i < len(doc_embeddings):
                # Semantic similarity scoring
                score = self._cosine_similarity(
                    query_embedding, doc_embeddings[i]
                )
            else:
                # Fallback: keyword overlap scoring
                score = self._keyword_overlap_score(query, doc)

            confidence_level = self._get_confidence_level(score)
            ranked.append(RankedDocument(
                content=doc,
                relevance_score=round(score, 4),
                source=f"document_{i+1}",
                confidence_level=confidence_level,
            ))

        ranked.sort(key=lambda x: x.relevance_score, reverse=True)
        logger.info(
            f"Reranked {len(ranked)} documents. "
            f"Top score: {ranked[0].relevance_score:.3f}"
        )
        return ranked

    def get_crag_strategy(self, ranked_docs: list[RankedDocument]) -> dict:
        """
        Determines retrieval strategy based on CRAG confidence evaluation.

        Args:
            ranked_docs: Reranked documents with scores

        Returns:
            dict with strategy, confidence, and recommendation
        """
        if not ranked_docs:
            return {
                "strategy": "no_docs",
                "confidence": 0.0,
                "recommendation": "No documents available. Use alternative source.",
                "use_rag": False,
                "use_web": True,
            }

        avg_confidence = sum(d.relevance_score for d in ranked_docs) / len(ranked_docs)
        top_confidence = ranked_docs[0].relevance_score

        # Use top document confidence as primary signal
        primary_confidence = top_confidence

        if primary_confidence >= self.CONFIDENCE_HIGH:
            strategy = "rag_only"
            recommendation = "High confidence. Use RAG documents directly."
            use_rag = True
            use_web = False
        elif primary_confidence >= self.CONFIDENCE_MEDIUM:
            strategy = "rag_plus_web"
            recommendation = "Medium confidence. Combine RAG with supplementary search."
            use_rag = True
            use_web = True
        else:
            strategy = "web_only"
            recommendation = "Low confidence. Discard RAG documents, use alternative source."
            use_rag = False
            use_web = True

        return {
            "strategy": strategy,
            "confidence": round(primary_confidence, 4),
            "avg_confidence": round(avg_confidence, 4),
            "recommendation": recommendation,
            "use_rag": use_rag,
            "use_web": use_web,
            "top_documents": [
                {"content": d.content[:200], "score": d.relevance_score}
                for d in ranked_docs[:3]
            ],
        }

    def filter_by_confidence(
        self,
        ranked_docs: list[RankedDocument],
        min_score: float = 0.3,
    ) -> list[RankedDocument]:
        """Filters out documents below minimum confidence threshold."""
        filtered = [d for d in ranked_docs if d.relevance_score >= min_score]
        logger.info(
            f"Filtered documents: {len(ranked_docs)} → {len(filtered)} "
            f"(min_score={min_score})"
        )
        return filtered

    # ── PRIVATE HELPERS ──────────────────────────────

    def _cosine_similarity(
        self, vec_a: list[float], vec_b: list[float]
    ) -> float:
        """Computes cosine similarity between two vectors."""
        a = np.array(vec_a)
        b = np.array(vec_b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _keyword_overlap_score(self, query: str, document: str) -> float:
        """
        Fallback scoring based on keyword overlap.
        Used when embeddings are not available.
        """
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())
        if not query_words:
            return 0.0
        overlap = query_words.intersection(doc_words)
        return len(overlap) / len(query_words)

    def _get_confidence_level(self, score: float) -> str:
        if score >= self.CONFIDENCE_HIGH:
            return "high"
        elif score >= self.CONFIDENCE_MEDIUM:
            return "medium"
        return "low"
