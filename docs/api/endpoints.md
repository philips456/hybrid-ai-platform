# API Endpoints Documentation

# Hybrid AI Platform — PFE Nomo Philippe André

# ESPRIT School of Engineering / Philipps-Universität Marburg

Base URL : http://localhost:8000
Documentation interactive : http://localhost:8000/docs

---

## Authentification

POST /auth/token
Body : {"username": "analyst", "password": "analyst123"}
Retourne : {"access_token": "...", "token_type": "bearer"}
Rôles disponibles : admin | analyst | viewer

---

## Health

GET /health
Auth : Non requise
Retourne : {"status": "ok", "version": "1.0.0", "domain": "telecom"}

GET /ready
Auth : Non requise
Vérifie : PostgreSQL + Qdrant + MLflow
Retourne : {"status": "ok|degraded", "services": [...]}

---

## Prédictions

GET /predictions
Auth : viewer+
Params : series_id (optionnel), limit (max 500)
Retourne : Liste des prédictions CNN+LSTM

GET /predictions/summary/metrics
Auth : viewer+
Retourne : {"rmse": float, "mae": float, "r2": float, "model_status": str}

---

## Anomalies

GET /anomalies
Auth : viewer+
Params : status (detected|confirmed|rejected), limit (max 500)
Retourne : Liste des anomalies avec scores

GET /anomalies/{anomaly_id}
Auth : viewer+
Retourne : Détail d'une anomalie

PUT /anomalies/{anomaly_id}/status
Auth : analyst+
Body : {"status": "confirmed|rejected", "reason": "..."}
Retourne : Anomalie mise à jour

GET /anomalies/summary/stats
Auth : viewer+
Retourne : {"total_detected": int, "confirmed": int, "false_positive_rate": float}

---

## Agents LLM

POST /agents/run
Auth : analyst+
Body : {
"task": "feedback|analyse|research|report",
"domain": "synthetic|telecom|finance|industry",
"metrics": {"rmse": float, "consecutive_periods": int, "mae": float},
"model_config_override": {...}
}
Retourne : {
"run_id": str,
"status": "completed|failed",
"task": str,
"result": {
"feedback_result": {"triggered": bool, "suggestions": [...]},
"analysis_result": {"suggestions": [...]},
"research_result": {"synthesis": {...}, "confidence": float}
},
"latency_ms": int
}
Note : Timeout recommandé 120s — appel LLM Claude inclus

GET /agents/runs
Auth : viewer+
Retourne : Historique des runs d'agents

---

## Feedback Loop HITL

GET /feedback/pending
Auth : viewer+
Retourne : Liste des suggestions en attente de validation HITL
Format : [{
"id": uuid,
"hyperparameter": str,
"current_value": str,
"suggested_value": str,
"justification": str,
"confidence_score": float,
"trigger_metrics": {"rmse": float, "consecutive_periods": int},
"status": "PENDING",
"created_at": datetime
}]

POST /feedback/{suggestion_id}/validate
Auth : analyst+
Body : {"approved": bool, "reason": "..."}
Retourne : {
"status": "APPROVED|REJECTED",
"decided_by": str,
"next_action": "Model retraining scheduled|No action taken",
"db_updated": bool
}
Note : Rejection reason stored in LongTermMemory — not reproposed to Claude

GET /feedback/history
Auth : viewer+
Retourne : Toutes les suggestions traitées (APPROVED + REJECTED)

GET /feedback/stats
Auth : viewer+
Retourne : {"pending": int, "approved": int, "rejected": int, "approval_rate": float}

---

## Rapports

GET /reports
Auth : viewer+
Retourne : Liste des rapports générés par ReporterAgent

GET /reports/{report_id}
Auth : viewer+
Retourne : Rapport complet avec scores LLM-as-a-judge
Format : {
"title": str,
"summary": str,
"findings": [{"finding": str, "evidence": str, "severity": str}],
"recommendations": [{"action": str, "rationale": str, "priority": str}],
"faithfulness_score": float,
"relevance_score": float,
"\_quality_score": float,
"\_approved": bool
}

---

## Sécurité

Tous les endpoints (sauf /health, /ready, /auth/token) requièrent :
Header : Authorization: Bearer <JWT>

Rôles :
viewer — lecture seule
analyst — lecture + validation HITL + run agents
admin — accès complet

HMAC : Chaque suggestion est signée avec HMAC-SHA-256
Tampering détecté automatiquement avant validation HITL

Data Privacy : Métriques anonymisées avant envoi à Claude API
Basé sur GDPR Article 25 + Huang et al. arXiv:2410.11182

---

## Codes de réponse

200 — Succès
401 — Token invalide ou expiré
403 — Rôle insuffisant
404 — Ressource non trouvée
422 — Données invalides ou suggestion déjà traitée
500 — Erreur serveur interne
