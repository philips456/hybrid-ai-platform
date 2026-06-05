"""
src/dashboard/pages/06_mlflow.py
MLflow Experiments page — real data from MLflow API.
"""
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="MLflow", page_icon="📈", layout="wide")

if "token" not in st.session_state:
    st.warning("Please login from the main page.")
    st.stop()

st.title("📈 MLflow Experiments")
st.caption("Real training runs — CNN+LSTM+Attention fraud detection model")

MLFLOW_URL = "http://localhost:5000"


def parse_metrics(run):
    """MLflow returns metrics as list [{key, value, step}] — convert to dict."""
    metrics_list = run.get("data", {}).get("metrics", [])
    if isinstance(metrics_list, list):
        return {m["key"]: m["value"] for m in metrics_list}
    return metrics_list  # already dict fallback


def parse_params(run):
    """MLflow returns params as list [{key, value}] — convert to dict."""
    params_list = run.get("data", {}).get("params", [])
    if isinstance(params_list, list):
        return {p["key"]: p["value"] for p in params_list}
    return params_list


def get_mlflow_runs():
    try:
        resp = requests.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/runs/search",
            json={
                "experiment_ids": ["1"],
                "order_by": ["start_time DESC"],
                "max_results": 20,
            },
            timeout=5
        )
        if resp.status_code == 200:
            return resp.json().get("runs", [])
        return []
    except Exception:
        return []


def get_run_metrics_history(run_id, metric_key):
    try:
        resp = requests.get(
            f"{MLFLOW_URL}/api/2.0/mlflow/metrics/get-history",
            params={"run_id": run_id, "metric_key": metric_key},
            timeout=5
        )
        if resp.status_code == 200:
            return resp.json().get("metrics", [])
        return []
    except Exception:
        return []


# ── STATUS ────────────────────────────────────────────
try:
    resp = requests.get(f"{MLFLOW_URL}/health", timeout=3)
    mlflow_ok = resp.status_code == 200
except Exception:
    mlflow_ok = False

if mlflow_ok:
    st.success(f"MLflow running — [Open MLflow UI]({MLFLOW_URL})")
else:
    st.warning("MLflow not reachable at localhost:5000")

col1, col2, col3 = st.columns(3)
with col1:
    st.link_button("🔗 Open MLflow UI", MLFLOW_URL,
                   use_container_width=True, type="primary")
with col2:
    st.link_button("📊 Experiments", f"{MLFLOW_URL}/#/experiments/1",
                   use_container_width=True)
with col3:
    st.link_button("🏆 Best Run", f"{MLFLOW_URL}/#/experiments/1",
                   use_container_width=True)

st.divider()

# ── LOAD RUNS ─────────────────────────────────────────
runs = get_mlflow_runs()

if not runs:
    st.warning("No runs found — make sure MLflow is running and notebook 03 has been executed.")
else:
    st.subheader(f"Training Runs ({len(runs)} total)")

    # Build summary table
    run_data = []
    for run in runs:
        info    = run.get("info", {})
        metrics = parse_metrics(run)
        params  = parse_params(run)

        duration_ms  = (info.get("end_time", 0) - info.get("start_time", 0))
        duration_min = duration_ms / 60000 if duration_ms > 0 else 0

        run_data.append({
            "Run ID":   info.get("run_id", "")[:8] + "...",
            "Status":   info.get("status", ""),
            "AUC-ROC":  f"{metrics.get('best_val_auc', metrics.get('test_auc_roc', 0)):.4f}",
            "F1":       f"{metrics.get('test_f1', 0):.4f}",
            "RMSE":     f"{metrics.get('test_rmse', 0):.4f}",
            "LR":       params.get("learning_rate", "N/A"),
            "SMOTE":    params.get("smote_strategy", "N/A"),
            "Duration": f"{duration_min:.1f} min",
        })

    df = pd.DataFrame(run_data)
    st.dataframe(df, use_container_width=True)

    st.divider()

    # ── BEST RUN ──────────────────────────────────────
    best_run     = runs[0]
    best_metrics = parse_metrics(best_run)
    best_params  = parse_params(best_run)
    best_run_id  = best_run.get("info", {}).get("run_id", "")

    st.subheader("Best Run — Detailed Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AUC-ROC", f"{best_metrics.get('best_val_auc', best_metrics.get('test_auc_roc', 0)):.4f}")
    col2.metric("F1-score", f"{best_metrics.get('test_f1', 0):.4f}")
    col3.metric("RMSE",     f"{best_metrics.get('test_rmse', 0):.4f}")
    col4.metric("Best Epoch", f"{int(best_metrics.get('best_epoch', 0))}")

    st.divider()

    # ── TRAINING HISTORY ──────────────────────────────
    st.subheader("Training History — Real Data from MLflow")

    col1, col2 = st.columns(2)

    with col1:
        auc_history = get_run_metrics_history(best_run_id, "val_auc")
        if auc_history:
            steps  = [m["step"] for m in auc_history]
            values = [m["value"] for m in auc_history]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=steps, y=values, mode='lines+markers',
                name='Val AUC', line=dict(color='#2196F3', width=2)
            ))
            fig.add_hline(y=0.90, line_dash="dash", line_color="green",
                          annotation_text="Target: 0.90")
            fig.update_layout(
                title="Validation AUC-ROC per Epoch",
                xaxis_title="Epoch", yaxis_title="AUC-ROC",
                height=300,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("AUC history not available for this run")

    with col2:
        loss_history = get_run_metrics_history(best_run_id, "val_loss")
        if loss_history:
            steps  = [m["step"] for m in loss_history]
            values = [m["value"] for m in loss_history]
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=steps, y=values, mode='lines+markers',
                name='Val Loss', line=dict(color='#F44336', width=2)
            ))
            fig2.update_layout(
                title="Validation Loss per Epoch",
                xaxis_title="Epoch", yaxis_title="Loss",
                height=300,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Loss history not available for this run")

    st.divider()

    # ── RUNS COMPARISON ───────────────────────────────
    if len(runs) > 1:
        st.subheader("Runs Comparison")
        run_names = [f"Run {i+1}" for i in range(len(runs))]

        auc_vals = []
        f1_vals  = []
        for r in runs:
            m = parse_metrics(r)
            auc_vals.append(float(m.get("best_val_auc", m.get("test_auc_roc", 0))))
            f1_vals.append(float(m.get("test_f1", 0)))

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=run_names, y=auc_vals,
            name='AUC-ROC', marker_color='#2196F3', opacity=0.8
        ))
        fig3.add_trace(go.Bar(
            x=run_names, y=f1_vals,
            name='F1-score', marker_color='#FF9800', opacity=0.8
        ))
        fig3.add_hline(y=0.90, line_dash="dash", line_color="green",
                       annotation_text="Target AUC: 0.90")
        fig3.update_layout(
            title="All Runs Comparison — AUC-ROC vs F1-score",
            barmode='group', height=350,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig3, use_container_width=True)

    best_auc = max(auc_vals) if len(runs) > 1 else float(
        best_metrics.get("best_val_auc", best_metrics.get("test_auc_roc", 0))
    )
    st.caption(
        f"MLflow experiment: fraud_detection_cnn_lstm | "
        f"{len(runs)} runs | "
        f"Best AUC: {best_auc:.4f}"
    )