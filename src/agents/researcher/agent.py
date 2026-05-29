"""
src/agents/researcher/agent.py
ResearcherAgent — retrieves and synthesizes information from Qdrant.
Implements CRAG strategy for reliable retrieval.
"""
import json
import logging

import anthropic

from configs.settings import settings
from src.agents.context.builder import ContextBuilder
from src.agents.researcher.prompts import RESEARCHER_SYSTEM_PROMPT, RESEARCHER_QUERY_REFINEMENT_PROMPT
from src.agents.researcher.rag import RAGPipeline

logger = logging.getLogger(__name__)


class ResearcherAgent:
    """
    Retrieves relevant information from the knowledge base using CRAG.

    Usage:
        agent = ResearcherAgent(domain="telecom")
        result = agent.research("What causes high RMSE in 5G throughput prediction?")
    """

    def __init__(self, domain: str = "synthetic"):
        self.domain = domain
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.rag = RAGPipeline()
        self.context_builder = ContextBuilder(agent_type="researcher")
        self.model = settings.anthropic_model

    def research(self, query: str, max_retries: int = 2) -> dict:
        """
        Researches a query using RAG with CRAG confidence evaluation.

        Args:
            query: Research question
            max_retries: Max query reformulation attempts if confidence is low

        Returns:
            dict with synthesis, confidence, and source information
        """
        current_query = query
        rag_result = None

        for attempt in range(max_retries + 1):
            rag_result = self.rag.retrieve(current_query)

            if rag_result["confidence"] >= 0.4 or attempt == max_retries:
                break

            # Low confidence — reformulate query
            logger.info(
                f"ResearcherAgent: low confidence ({rag_result['confidence']:.2f}), "
                f"reformulating query (attempt {attempt + 1})"
            )
            current_query = self._reformulate_query(query, rag_result["confidence"])

        # Build context and synthesize
        context = self.context_builder.build_researcher_context(
            query=current_query,
            rag_documents=rag_result["documents"],
            confidence_scores=[rag_result["confidence"]],
            web_results=None,
        )

        system_prompt = RESEARCHER_SYSTEM_PROMPT.format(domain=self.domain)
        synthesis = self._synthesize(system_prompt, context.build())

        return {
            "query": query,
            "synthesis": synthesis,
            "confidence": rag_result["confidence"],
            "strategy": rag_result["strategy"],
            "documents_used": len(rag_result["documents"]),
        }

    def _reformulate_query(self, original_query: str, confidence: float) -> str:
        """Reformulates a low-confidence query using the LLM."""
        prompt = RESEARCHER_QUERY_REFINEMENT_PROMPT.format(
            original_query=original_query,
            confidence=confidence,
        )
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=256,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.error(f"Query reformulation failed: {e}")
            return original_query

    def _synthesize(self, system: str, human: str) -> dict:
        """Calls LLM to synthesize retrieved information."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.1,
                system=system,
                messages=[{"role": "user", "content": human}],
            )
            content = response.content[0].text.strip()
            if "```" in content:
               lines = content.split("\n")
               lines = [l for l in lines if not l.strip().startswith("```")]
               content = "\n".join(lines).strip()
            return json.loads(content)
        except Exception as e:
            logger.error(f"ResearcherAgent synthesis failed: {e}")
            return {"synthesis": "Research failed.", "confidence": 0.0, "gaps": []}
