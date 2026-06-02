"""
src/agents/analyst/agent.py
AnalystAgent — analyzes DL model residuals and generates feedback suggestions.

Implements:
- Anthropic Tool Use (Dang et al., arXiv:2509.18076) — 99.8% schema compliance
- detect_anomaly_type before suggest (SAGE, arXiv:2605.05725)
- SuggestionValidator — validates bounds and confidence before HITL
- Reflexion pattern (Singh et al., arXiv:2501.09136)
- Context Engineering (Zhang et al., arXiv:2510.04618)
"""
import json
import logging

import anthropic

from configs.settings import settings
from src.agents.analyst.prompts import ANALYST_SYSTEM_PROMPT, ANALYST_REFLEXION_PROMPT
from src.agents.analyst.tools import ANALYST_TOOLS, FEEDBACK_SUGGESTION_TOOL
from src.agents.analyst.validator import SuggestionValidator
from src.agents.context.builder import ContextBuilder

logger = logging.getLogger(__name__)


class AnalystAgent:
    """
    Analyzes CNN+LSTM model residuals and generates hyperparameter suggestions.

    Pipeline:
    1. detect_anomaly_type — classify anomaly (SAGE arXiv:2605.05725)
    2. submit_feedback_suggestion — propose adjustments via Tool Use
    3. SuggestionValidator — validate bounds and confidence
    4. Optional Reflexion — self-critique

    Performance note:
    detect_anomaly_type adds ~30s (1 extra LLM call).
    Set enable_classification=False for faster demo responses.
    """

    def __init__(self, domain: str = "synthetic", enable_classification: bool = False):
        self.domain = domain
        self.enable_classification = enable_classification
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.context_builder = ContextBuilder(agent_type="analyst")
        self.validator = SuggestionValidator(min_confidence=0.5, strict_mode=False)
        self.model = settings.anthropic_model

    def analyze(
        self,
        metrics: dict,
        model_config: dict,
        rag_documents: list[str] = None,
        anomaly_history: list[dict] = None,
        previous_suggestions: list[dict] = None,
        use_reflexion: bool = False,
    ) -> list[dict]:
        """
        Analyzes metrics and generates validated feedback suggestions.

        Returns:
            List of validated FeedbackSuggestion dicts
        """
        rag_documents = rag_documents or []
        anomaly_history = anomaly_history or []
        previous_suggestions = previous_suggestions or []

        context = self.context_builder.build_analyst_context(
            metrics=metrics,
            model_config=model_config,
            rag_documents=rag_documents,
            anomaly_history=anomaly_history,
            previous_suggestions=previous_suggestions,
        )

        system_prompt = ANALYST_SYSTEM_PROMPT.format(domain=self.domain)

        # Step 1: Classify anomaly type (optional — adds ~30s)
        classification = {}
        if self.enable_classification:
            logger.info("AnalystAgent: classifying anomaly type")
            classification = self._detect_anomaly_type(
                system=system_prompt,
                human=context.build(),
            )
            if classification:
                logger.info(
                    f"AnalystAgent: type={classification.get('anomaly_type')} "
                    f"severity={classification.get('severity')} "
                    f"root_cause={classification.get('root_cause_hypothesis', '')[:60]}"
                )
        else:
            logger.info("AnalystAgent: classification disabled — skipping detect_anomaly_type")

        # Step 2: Generate suggestions via Tool Use
        logger.info("AnalystAgent: generating suggestions via Tool Use")
        enriched = self._enrich_context_with_classification(context.build(), classification)
        raw_suggestions = self._call_llm_with_tools(system=system_prompt, human=enriched)

        # Step 3: Validate — add metadata, keep all suggestions for HITL
        validation_results = self.validator.validate_all(raw_suggestions)
        enriched_suggestions = []
        for result in validation_results:
            s = result.suggestion.copy()
            s["validation_issues"] = result.issues
            s["validation_warnings"] = result.warnings
            s["pre_validated"] = result.is_valid
            s["anomaly_classification"] = classification
            enriched_suggestions.append(s)

        valid_count = sum(1 for r in validation_results if r.is_valid)
        logger.info(
            f"AnalystAgent: {len(raw_suggestions)} generated, "
            f"{valid_count} pre-validated, all sent to HITL"
        )

        if not enriched_suggestions or not use_reflexion:
            return enriched_suggestions

        # Step 4: Reflexion self-critique
        logger.info("AnalystAgent: applying Reflexion self-critique")
        reflexion_prompt = ANALYST_REFLEXION_PROMPT.format(
            previous_suggestions=json.dumps(enriched_suggestions, indent=2)
        )
        refined = self._call_llm_with_tools(system=system_prompt, human=reflexion_prompt)
        final_results = self.validator.validate_all(refined)
        final = []
        for result in final_results:
            s = result.suggestion.copy()
            s["validation_issues"] = result.issues
            s["validation_warnings"] = result.warnings
            s["pre_validated"] = result.is_valid
            s["anomaly_classification"] = classification
            final.append(s)

        logger.info(f"AnalystAgent: {len(final)} suggestions after Reflexion")
        return final

    def _detect_anomaly_type(self, system: str, human: str) -> dict:
        """Classifies anomaly type before generating suggestions."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                temperature=0.0,
                system=system,
                tools=ANALYST_TOOLS,
                tool_choice={"type": "tool", "name": "detect_anomaly_type"},
                messages=[{
                    "role": "user",
                    "content": (
                        f"{human}\n\n"
                        "Classify the anomaly type using detect_anomaly_type tool."
                    )
                }],
            )
            for block in response.content:
                if block.type == "tool_use" and block.name == "detect_anomaly_type":
                    return dict(block.input)
            return {}
        except Exception as e:
            logger.warning(f"AnalystAgent: anomaly classification failed: {e}")
            return {}

    def _call_llm_with_tools(self, system: str, human: str) -> list[dict]:
        """Calls Claude with Tool Use — tool_choice=any forces at least one call."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                temperature=0.1,
                system=system,
                tools=[FEEDBACK_SUGGESTION_TOOL],
                tool_choice={"type": "any"},
                messages=[{"role": "user", "content": human}],
            )
            suggestions = []
            for block in response.content:
                if block.type == "tool_use" and block.name == "submit_feedback_suggestion":
                    suggestion = dict(block.input)
                    suggestion["confidence_score"] = max(
                        0.0, min(1.0, float(suggestion.get("confidence_score", 0.5)))
                    )
                    suggestions.append(suggestion)
                    logger.debug(
                        f"AnalystAgent: tool_use → {suggestion['hyperparameter']} "
                        f"confidence={suggestion['confidence_score']:.2f}"
                    )
            logger.info(
                f"AnalystAgent: {len(suggestions)} raw suggestion(s) "
                f"(stop_reason={response.stop_reason})"
            )
            return suggestions
        except Exception as e:
            logger.error(f"AnalystAgent: Tool Use failed: {e}")
            return []

    def _enrich_context_with_classification(self, context: str, classification: dict) -> str:
        """Enriches context with anomaly classification."""
        if not classification:
            return context
        return context + (
            f"\n\nANOMALY CLASSIFICATION:\n"
            f"  Type: {classification.get('anomaly_type', 'unknown')}\n"
            f"  Severity: {classification.get('severity', 'unknown')}\n"
            f"  Pattern: {classification.get('pattern_description', '')}\n"
            f"  Root cause: {classification.get('root_cause_hypothesis', '')}\n"
        )