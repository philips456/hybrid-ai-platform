"""
src/agents/analyst/tools.py
Tool definitions for AnalystAgent.

Includes detect_anomaly_type tool based on:
- SAGE (Kang et al., arXiv:2605.05725) — specialized analyzers per anomaly type
- TSAD-Agents (Xu et al., arXiv:2502.17812) — dynamic tool set selection
"""

FEEDBACK_SUGGESTION_TOOL = {
    "name": "submit_feedback_suggestion",
    "description": (
        "Submit a hyperparameter adjustment suggestion based on model performance analysis. "
        "Call once per suggested hyperparameter change. Maximum 3 calls."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "hyperparameter": {
                "type": "string",
                "description": "Name of the hyperparameter (e.g. learning_rate, window_size, dropout_rate, batch_size)",
            },
            "current_value": {
                "type": "string",
                "description": "Current value as string",
            },
            "suggested_value": {
                "type": "string",
                "description": "Suggested new value as string",
            },
            "justification": {
                "type": "string",
                "description": "Data-driven justification referencing specific metrics",
            },
            "confidence_score": {
                "type": "number",
                "description": "Confidence between 0.0 and 1.0",
            },
        },
        "required": [
            "hyperparameter",
            "current_value",
            "suggested_value",
            "justification",
            "confidence_score",
        ],
    },
}

DETECT_ANOMALY_TYPE_TOOL = {
    "name": "detect_anomaly_type",
    "description": (
        "Classify the type of anomaly before suggesting hyperparameter adjustments. "
        "Based on SAGE (arXiv:2605.05725) — specialized analysis per anomaly type. "
        "Call this FIRST before submit_feedback_suggestion."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "anomaly_type": {
                "type": "string",
                "description": "point | contextual | collective | trend",
            },
            "pattern_description": {
                "type": "string",
                "description": "Brief description of the detected pattern",
            },
            "affected_periods": {
                "type": "integer",
                "description": "Number of periods affected by the anomaly",
            },
            "severity": {
                "type": "string",
                "description": "low | medium | high | critical",
            },
            "root_cause_hypothesis": {
                "type": "string",
                "description": "Most likely root cause based on the metrics",
            },
        },
        "required": [
            "anomaly_type",
            "pattern_description",
            "affected_periods",
            "severity",
            "root_cause_hypothesis",
        ],
    },
}

# Tool set for AnalystAgent — ordered by execution sequence
ANALYST_TOOLS = [
    DETECT_ANOMALY_TYPE_TOOL,    # Step 1 — classify anomaly
    FEEDBACK_SUGGESTION_TOOL,    # Step 2 — suggest adjustments
]
