"""
src/agents/analyst/agent.py
AnalystAgent — analyzes DL model residuals and generates feedback suggestions.

Implements:
- Reflexion pattern (Singh et al., arXiv:2501.09136)
- Context Engineering (Zhang et al., arXiv:2510.04618)
"""
import json
import logging
from typing import Optional

import anthropic

from configs.settings import settings
from src.agents.analyst.prompts import ANALYST_SYSTEM_PROMPT, ANALYST_REFLEXION_PROMPT
from src.agents.context.builder import ContextBuilder

logger = logging.getLogger(__name__)


class AnalystAgent:
    """
    Analyzes CNN+LSTM model residuals and generates hyperparameter suggestions.

    Reflexion pattern:
    1. Generate initial analysis
    2. Critique own suggestions
    3. Refine if needed
    4. Submit final suggestions for HITL validation

    Usage:
        agent = AnalystAgent(domain="telecom")
        suggestions = agent.analyze(
            metrics={"rmse": 0.18, "consecutive_periods": 6},
            model_config={...},
            rag_documents=[...],
            anomaly_history=[...]
        )
    """

    def __init__(self, domain: str = "synthetic"):
        self.domain = domain
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.context_builder = ContextBuilder(agent_type="analyst")
        self.model = settings.anthropic_model

    def analyze(
        self,
        metrics: dict,
        model_config: dict,
        rag_documents: list[str] = None,
        anomaly_history: list[dict] = None,
        previous_suggestions: list[dict] = None,
        use_reflexion: bool = True,
    ) -> list[dict]:
        """
        Analyzes metrics and generates feedback suggestions.

        Args:
            metrics: Current model metrics (rmse, mae, consecutive_periods)
            model_config: Model configuration with thresholds
            rag_documents: Relevant documents from Qdrant
            anomaly_history: Recent anomaly events
            previous_suggestions: Previous HITL feedback suggestions
            use_reflexion: Whether to apply Reflexion self-critique

        Returns:
            List of FeedbackSuggestion dicts
        """
        rag_documents = rag_documents or []
        anomaly_history = anomaly_history or []
        previous_suggestions = previous_suggestions or []

        # Build context using Context Engineering
        context = self.context_builder.build_analyst_context(
            metrics=metrics,
            model_config=model_config,
            rag_documents=rag_documents,
            anomaly_history=anomaly_history,
            previous_suggestions=previous_suggestions,
        )

        system_prompt = ANALYST_SYSTEM_PROMPT.format(domain=self.domain)

        # Step 1: Initial analysis
        logger.info("AnalystAgent: generating initial analysis")
        initial_suggestions = self._call_llm(
            system=system_prompt,
            human=context.build(),
        )

        if not initial_suggestions or not use_reflexion:
            return initial_suggestions

        # Step 2: Reflexion — self-critique
        logger.info("AnalystAgent: applying Reflexion self-critique")
        reflexion_prompt = ANALYST_REFLEXION_PROMPT.format(
            previous_suggestions=json.dumps(initial_suggestions, indent=2)
        )

        final_suggestions = self._call_llm(
            system=system_prompt,
            human=reflexion_prompt,
        )

        logger.info(
            f"AnalystAgent: {len(final_suggestions)} suggestions after Reflexion"
        )
        return final_suggestions

    def _clean_json(self, content: str) -> str:
        """Removes markdown backticks from LLM response."""
        content = content.strip()
        if "```" in content:
            lines = content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines).strip()
        return content

    def _call_llm(self, system: str, human: str) -> list[dict]:
        """Calls Claude with prefill to force JSON array output."""
        content = ""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                temperature=0.1,
                system=system,
                messages=[
                    {"role": "user", "content": human},
                    {"role": "assistant", "content": "["},
                ],
            )
            # Prepend the prefill bracket back
            content = "[" + self._clean_json(response.content[0].text)
            suggestions = json.loads(content)
            if not isinstance(suggestions, list):
                suggestions = [suggestions]
            return suggestions

        except json.JSONDecodeError as e:
            logger.error(f"AnalystAgent: JSON parse error: {e}")
            logger.error(f"AnalystAgent: raw content: {content[:300]}")
            return []
        except Exception as e:
            logger.error(f"AnalystAgent: LLM call failed: {e}")
            return []