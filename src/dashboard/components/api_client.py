"""
src/dashboard/components/api_client.py
HTTP client for Streamlit dashboard → FastAPI communication.
"""
import requests
import streamlit as st

import os
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")


def get_token() -> str:
    """Returns cached JWT token from session state."""
    return st.session_state.get("token", "")


def get_headers() -> dict:
    return {"Authorization": f"Bearer {get_token()}"}


def login(username: str, password: str) -> bool:
    """Authenticates and stores token in session state."""
    try:
        resp = requests.post(
            f"{API_BASE}/auth/token",
            json={"username": username, "password": password},
            timeout=5,
        )
        if resp.status_code == 200:
            st.session_state["token"] = resp.json()["access_token"]
            st.session_state["username"] = username
            return True
        return False
    except Exception:
        return False


def get_health() -> dict:
    try:
        resp = requests.get(f"{API_BASE}/ready", timeout=5)
        return resp.json()
    except Exception:
        return {"status": "down", "services": []}


def get_metrics() -> dict:
    try:
        resp = requests.get(
            f"{API_BASE}/predictions/summary/metrics",
            headers=get_headers(), timeout=5
        )
        return resp.json() if resp.status_code == 200 else {}
    except Exception:
        return {}


def get_anomalies(status: str = None) -> list:
    try:
        params = {"limit": 100}
        if status:
            params["status"] = status
        resp = requests.get(
            f"{API_BASE}/anomalies",
            headers=get_headers(), params=params, timeout=5
        )
        return resp.json() if resp.status_code == 200 else []
    except Exception:
        return []


def get_anomaly_stats() -> dict:
    try:
        resp = requests.get(
            f"{API_BASE}/anomalies/summary/stats",
            headers=get_headers(), timeout=5
        )
        return resp.json() if resp.status_code == 200 else {}
    except Exception:
        return {}


def get_pending_feedback() -> list:
    try:
        resp = requests.get(
            f"{API_BASE}/feedback/pending",
            headers=get_headers(), timeout=5
        )
        return resp.json() if resp.status_code == 200 else []
    except Exception:
        return []


def get_feedback_stats() -> dict:
    try:
        resp = requests.get(
            f"{API_BASE}/feedback/stats",
            headers=get_headers(), timeout=5
        )
        return resp.json() if resp.status_code == 200 else {}
    except Exception:
        return {}


def validate_feedback(suggestion_id: str, approved: bool, reason: str = "") -> dict:
    try:
        resp = requests.post(
            f"{API_BASE}/feedback/{suggestion_id}/validate",
            headers=get_headers(),
            json={"approved": approved, "reason": reason},
            timeout=10,
        )
        return resp.json() if resp.status_code == 200 else {"error": resp.text}
    except Exception as e:
        return {"error": str(e)}


def run_agent(task: str, metrics: dict = None) -> dict:
    try:
        resp = requests.post(
            f"{API_BASE}/agents/run",
            headers=get_headers(),
            json={"task": task, "metrics": metrics or {}},
            timeout=120,
        )
        return resp.json() if resp.status_code == 200 else {"error": resp.text}
    except Exception as e:
        return {"error": str(e)}


def get_reports() -> list:
    try:
        resp = requests.get(
            f"{API_BASE}/reports",
            headers=get_headers(), timeout=5
        )
        return resp.json() if resp.status_code == 200 else []
    except Exception:
        return []
