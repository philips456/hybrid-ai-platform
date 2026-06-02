"""
tests/unit/test_agents/test_tools.py
Unit tests for Tool Use schemas and detect_anomaly_type.
"""
import pytest
from src.agents.analyst.tools import (
    FEEDBACK_SUGGESTION_TOOL,
    DETECT_ANOMALY_TYPE_TOOL,
    ANALYST_TOOLS,
)


class TestToolSchemas:

    def test_feedback_suggestion_tool_has_required_fields(self):
        required = FEEDBACK_SUGGESTION_TOOL["input_schema"]["required"]
        assert "hyperparameter" in required
        assert "current_value" in required
        assert "suggested_value" in required
        assert "justification" in required
        assert "confidence_score" in required

    def test_detect_anomaly_type_tool_has_required_fields(self):
        required = DETECT_ANOMALY_TYPE_TOOL["input_schema"]["required"]
        assert "anomaly_type" in required
        assert "severity" in required
        assert "root_cause_hypothesis" in required

    def test_analyst_tools_order(self):
        """detect_anomaly_type must come before submit_feedback_suggestion."""
        names = [t["name"] for t in ANALYST_TOOLS]
        assert names.index("detect_anomaly_type") < names.index("submit_feedback_suggestion")

    def test_confidence_score_is_number_type(self):
        props = FEEDBACK_SUGGESTION_TOOL["input_schema"]["properties"]
        assert props["confidence_score"]["type"] == "number"

    def test_anomaly_type_is_string(self):
        props = DETECT_ANOMALY_TYPE_TOOL["input_schema"]["properties"]
        assert props["anomaly_type"]["type"] == "string"

    def test_findings_is_array_in_report_tool(self):
        from src.agents.reporter.agent import SUBMIT_REPORT_TOOL
        props = SUBMIT_REPORT_TOOL["input_schema"]["properties"]
        assert props["findings"]["type"] == "array"
        assert props["recommendations"]["type"] == "array"

    def test_judge_tool_has_approved_field(self):
        from src.agents.reporter.agent import JUDGE_TOOL
        props = JUDGE_TOOL["input_schema"]["properties"]
        assert "approved" in props
        assert props["approved"]["type"] == "boolean"
