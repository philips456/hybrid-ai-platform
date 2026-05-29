"""
src/agents/researcher/rag.py
RAG pipeline with CRAG confidence evaluation.

Based on:
- Yan et al., arXiv:2401.15884 (CRAG)
- Singh et al., arXiv:2501.09136 (Agentic RAG)
"""
import logging
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI

from configs.settings import settings
from src.agents.context.reranker import DocumentReranker, RankedDocument

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    RAG pipeline with CRAG confidence-based retrieval strategy.

    Pipeline:
    1. Embed query (text-embedding-3-small)
    2. Retrieve top-k from Qdrant
    3. Rerank with DocumentReranker
    4. Apply CRAG strategy based on confidence
    5. Return ranked documents + strategy

    Usage:
        rag = RAGPipeline()
        result = rag.retrieve(query="anomaly in debit_download_mbps")
    """

    COLLECTION_NAME = "hybrid_ai_documents"
    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_DIM = 1536
    TOP_K = 10
    RETURN_K = 3

    def __init__(self):
        self.qdrant = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        self.openai = OpenAI(api_key=settings.anthropic_api_key)
        self.reranker = DocumentReranker()
        self._ensure_collection()

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        min_confidence: float = 0.3,
    ) -> dict:
        """
        Retrieves and ranks documents for a query.

        Args:
            query: Search query
            top_k: Number of documents to retrieve
            min_confidence: Minimum relevance score to include

        Returns:
            dict with documents, strategy, and confidence info
        """
        top_k = top_k or self.TOP_K

        # Step 1: Embed query
        query_embedding = self._embed(query)

        # Step 2: Retrieve from Qdrant
        raw_results = self._search_qdrant(query_embedding, top_k)

        if not raw_results:
            return {
                "documents": [],
                "strategy": "no_results",
                "confidence": 0.0,
                "crag_recommendation": "No documents found in knowledge base.",
            }

        # Step 3: Rerank
        documents = [r["content"] for r in raw_results]
        doc_embeddings = [r.get("embedding") for r in raw_results]

        ranked = self.reranker.rerank(
            query=query,
            documents=documents,
            query_embedding=query_embedding,
            doc_embeddings=[e for e in doc_embeddings if e],
        )

        # Step 4: Filter by confidence
        filtered = self.reranker.filter_by_confidence(ranked, min_score=min_confidence)

        # Step 5: CRAG strategy
        crag = self.reranker.get_crag_strategy(filtered)

        logger.info(
            f"RAG: query='{query[:50]}' | "
            f"retrieved={len(raw_results)} | "
            f"filtered={len(filtered)} | "
            f"strategy={crag['strategy']} | "
            f"confidence={crag['confidence']}"
        )

        return {
            "documents": [d.content for d in filtered[:self.RETURN_K]],
            "ranked_documents": filtered[:self.RETURN_K],
            "strategy": crag["strategy"],
            "confidence": crag["confidence"],
            "crag_recommendation": crag["recommendation"],
            "use_web": crag["use_web"],
        }

    def ingest(self, documents: list[dict]) -> int:
        """
        Ingests documents into Qdrant.

        Args:
            documents: List of {"content": str, "metadata": dict}

        Returns:
            Number of documents ingested
        """
        if not documents:
            return 0

        points = []
        for i, doc in enumerate(documents):
            content = doc.get("content", "")
            if not content:
                continue

            embedding = self._embed(content)
            points.append(PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "content": content,
                    "metadata": doc.get("metadata", {}),
                }
            ))

        self.qdrant.upsert(
            collection_name=self.COLLECTION_NAME,
            points=points,
        )

        logger.info(f"RAG: ingested {len(points)} documents into Qdrant")
        return len(points)

    def _embed(self, text: str) -> list[float]:
        """Embeds text using text-embedding-3-small."""
        response = self.openai.embeddings.create(
            model=self.EMBEDDING_MODEL,
            input=text[:8000],  # Token limit safety
        )
        return response.data[0].embedding

    def _search_qdrant(self, query_embedding: list[float], top_k: int) -> list[dict]:
        """Searches Qdrant for similar documents."""
        try:
            results = self.qdrant.search(
                collection_name=self.COLLECTION_NAME,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True,
            )
            return [
                {
                    "content": r.payload.get("content", ""),
                    "score": r.score,
                    "metadata": r.payload.get("metadata", {}),
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return []

    def _ensure_collection(self) -> None:
        """Creates Qdrant collection if it does not exist."""
        try:
            collections = self.qdrant.get_collections().collections
            names = [c.name for c in collections]
            if self.COLLECTION_NAME not in names:
                self.qdrant.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=self.EMBEDDING_DIM,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"RAG: created Qdrant collection '{self.COLLECTION_NAME}'")
        except Exception as e:
            logger.warning(f"RAG: could not verify collection: {e}")
