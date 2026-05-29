"""
src/agents/reporter/agent.py
ReporterAgent — generates and self-evaluates structured reports.
Implements LLM-as-a-judge (Aggarwal et al., KDD 2025).
"""
import json
import logging

import anthropic

from configs.settings import settings
from src.agents.context.builder import ContextBuilder
from src.agents.reporter.prompts import REPORTER_SYSTEM_PROMPT, REPORTER_JUDGE_PROMPT

logger = logging.getLogger(__name__)

MIN_QUALITY_SCORE = 0.8
MAX_REGENERATION_ATTEMPTS = 2


class ReporterAgent:
    """
    Generates structured reports and evaluates them via LLM-as-a-judge.

    If quality score < 0.8, the report is regenerated automatically.
    Maximum 2 regeneration attempts before returning best available report.

    Usage:
        agent = ReporterAgent(domain="telecom")
        report = agent.generate_report(
            analysis_result={...},
            research_result={...},
            anomaly_data={...},
            report_type="anomaly_analysis"
        )
    """

    def __init__(self, domain: str = "synthetic"):
        self.domain = domain
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.context_builder = ContextBuilder(agent_type="reporter")
        self.model = settings.anthropic_model

    def generate_report(
        self,
        analysis_result: dict,
        research_result: dict,
        anomaly_data: dict,
        report_type: str = "anomaly_analysis",
    ) -> dict:
        """
        Generates a report and validates it with LLM-as-a-judge.

        Args:
            analysis_result: Output from AnalystAgent
            research_result: Output from ResearcherAgent
            anomaly_data: Raw anomaly data
            report_type: Type of report to generate

        Returns:
            Validated report dict with quality scores
        """
        context = self.context_builder.build_reporter_context(
            analysis_result=analysis_result,
            research_result=research_result,
            anomaly_data=anomaly_data,
            report_type=report_type,
        )

        system_prompt = REPORTER_SYSTEM_PROMPT.format(domain=self.domain)
        context_text = context.build()

        best_report = None
        best_score = 0.0

        for attempt in range(MAX_REGENERATION_ATTEMPTS + 1):
            logger.info(f"ReporterAgent: generating report (attempt {attempt + 1})")

            report = self._generate(system_prompt, context_text)
            if not report:
                continue

            # LLM-as-a-judge evaluation
            evaluation = self._evaluate(report, context_text)
            overall_score = evaluation.get("overall", 0.0)

            logger.info(
                f"ReporterAgent: report quality score = {overall_score:.2f} "
                f"(faithfulness={evaluation.get('faithfulness', 0):.2f}, "
                f"relevance={evaluation.get('relevance', 0):.2f}, "
                f"completeness={evaluation.get('completeness', 0):.2f})"
            )

            if overall_score > best_score:
                best_score = overall_score
                best_report = report
                best_report["_evaluation"] = evaluation

            if overall_score >= MIN_QUALITY_SCORE:
                break

            if attempt < MAX_REGENERATION_ATTEMPTS:
                logger.info(
                    f"ReporterAgent: quality below threshold ({overall_score:.2f} < "
                    f"{MIN_QUALITY_SCORE}), regenerating"
                )
                context_text = self._add_improvement_hints(
                    context_text, evaluation.get("issues", [])
                )

        if best_report:
            best_report["_quality_score"] = best_score
            best_report["_approved"] = best_score >= MIN_QUALITY_SCORE

        return best_report or {"error": "Report generation failed", "_quality_score": 0.0}

    def _generate(self, system: str, human: str) -> dict:
        """Generates a report via LLM."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                temperature=0.1,
                system=system,
                messages=[{"role": "user", "content": human}],
            )
            content = response.content[0].text.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content)
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return {}

    def _evaluate(self, report: dict, context: str) -> dict:
        """Evaluates report quality using LLM-as-a-judge."""
        judge_prompt = REPORTER_JUDGE_PROMPT.format(
            report=json.dumps(report, indent=2),
            context=context[:3000],
        )
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                temperature=0.0,
                messages=[{"role": "user", "content": judge_prompt}],
            )
            content = response.content[0].text.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content)
        except Exception as e:
            logger.error(f"Report evaluation failed: {e}")
            return {"overall": 0.5, "approved": False, "issues": [str(e)]}

    def _add_improvement_hints(self, context: str, issues: list[str]) -> str:
        """Adds improvement hints to context for regeneration."""
        if not issues:
            return context
        hints = "\n\nIMPROVEMENT REQUIRED:\n" + "\n".join(f"- {i}" for i in issues)
        return context + hints
