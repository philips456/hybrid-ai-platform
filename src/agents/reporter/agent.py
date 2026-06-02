"""
src/agents/reporter/agent.py
ReporterAgent — generates structured reports using Anthropic Tool Use.

Implements:
- Tool Use for reliable structured output (Dang et al., arXiv:2509.18076)
- LLM-as-a-judge evaluation (Aggarwal et al., KDD 2025)
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

SUBMIT_REPORT_TOOL = {
    "name": "submit_report",
    "description": "Submit a structured analysis report based on findings.",
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Report title",
            },
            "report_type": {
                "type": "string",
                "description": "anomaly_analysis | performance | feedback_suggestion",
            },
            "summary": {
                "type": "string",
                "description": "2-3 sentence executive summary",
            },
            "findings": {
                "type": "array",
                "description": "List of findings with evidence and severity",
                "items": {
                    "type": "object",
                    "properties": {
                        "finding": {"type": "string"},
                        "evidence": {"type": "string"},
                        "severity": {"type": "string"},
                    },
                    "required": ["finding", "evidence", "severity"],
                },
            },
            "recommendations": {
                "type": "array",
                "description": "List of recommendations with rationale and priority",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "rationale": {"type": "string"},
                        "priority": {"type": "string"},
                    },
                    "required": ["action", "rationale", "priority"],
                },
            },
            "confidence": {
                "type": "number",
                "description": "Overall confidence score between 0.0 and 1.0",
            },
        },
        "required": ["title", "report_type", "summary", "findings", "recommendations", "confidence"],
    },
}

JUDGE_TOOL = {
    "name": "submit_evaluation",
    "description": "Submit quality evaluation scores for a report.",
    "input_schema": {
        "type": "object",
        "properties": {
            "faithfulness": {
                "type": "number",
                "description": "0.0-1.0: Are all claims grounded in the context?",
            },
            "relevance": {
                "type": "number",
                "description": "0.0-1.0: Does the report address the actual question?",
            },
            "completeness": {
                "type": "number",
                "description": "0.0-1.0: Are all key findings captured?",
            },
            "overall": {
                "type": "number",
                "description": "0.0-1.0: Overall quality score",
            },
            "approved": {
                "type": "boolean",
                "description": "True if overall >= 0.8",
            },
            "issues": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of issues found (empty if approved)",
            },
        },
        "required": ["faithfulness", "relevance", "completeness", "overall", "approved", "issues"],
    },
}


class ReporterAgent:
    """
    Generates structured reports using Tool Use + LLM-as-a-judge.
    Tool Use ensures 99.8% schema compliance — no JSON parsing needed.
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
        """Generates and validates a report via Tool Use + LLM-as-a-judge."""
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

            report = self._generate_with_tools(system_prompt, context_text)
            if not report:
                continue

            evaluation = self._evaluate_with_tools(report, context_text)
            overall_score = evaluation.get("overall", 0.0)

            logger.info(
                f"ReporterAgent: quality={overall_score:.2f} "
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
                issues = evaluation.get("issues", [])
                if issues:
                    context_text += "\n\nIMPROVEMENT REQUIRED:\n" + "\n".join(
                        f"- {i}" for i in issues
                    )

        if best_report:
            best_report["_quality_score"] = best_score
            best_report["_approved"] = best_score >= MIN_QUALITY_SCORE

        return best_report or {"error": "Report generation failed", "_quality_score": 0.0}

    def _generate_with_tools(self, system: str, human: str) -> dict:
        """Generates report via Tool Use — no JSON parsing."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                temperature=0.1,
                system=system,
                tools=[SUBMIT_REPORT_TOOL],
                tool_choice={"type": "any"},
                messages=[{"role": "user", "content": human}],
            )
            for block in response.content:
                if block.type == "tool_use" and block.name == "submit_report":
                    report = dict(block.input)
                    report["confidence"] = max(0.0, min(1.0, float(report.get("confidence", 0.5))))
                    logger.debug(f"ReporterAgent: report generated via Tool Use")
                    return report
            return {}
        except Exception as e:
            logger.error(f"ReporterAgent: generation failed: {e}")
            return {}

    def _evaluate_with_tools(self, report: dict, context: str) -> dict:
        """Evaluates report quality via LLM-as-a-judge using Tool Use."""
        judge_prompt = REPORTER_JUDGE_PROMPT.format(
            report=json.dumps(report, indent=2, default=str),
            context=context[:3000],
        )
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                temperature=0.0,
                tools=[JUDGE_TOOL],
                tool_choice={"type": "any"},
                messages=[{"role": "user", "content": judge_prompt}],
            )
            for block in response.content:
                if block.type == "tool_use" and block.name == "submit_evaluation":
                    return dict(block.input)
            return {"overall": 0.5, "approved": False, "issues": ["Evaluation failed"]}
        except Exception as e:
            logger.error(f"ReporterAgent: evaluation failed: {e}")
            return {"overall": 0.5, "approved": False, "issues": [str(e)]}
