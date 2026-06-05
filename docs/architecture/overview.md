# Architecture Overview

# Hybrid AI Platform — PFE Nomo Philippe André

# ESPRIT School of Engineering / Philipps-Universität Marburg

---

## Couches de la plateforme

┌─────────────────────────────────────────────────────────┐
│ SOURCES DE DONNÉES │
│ Télécom | Finance | Industrie | Synthétique │
└─────────────────────────┬───────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────┐
│ COUCHE 1 — DATA PIPELINE (CRISP-DM Ph.3) │
│ DataLoader → DataCleaner → FeatureEngineer │
│ DataNormalizer → SequenceBuilder (window_size=24) │
│ Anti-leakage | IQR outliers | Cyclical encoding │
└─────────────────────────┬───────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────┐
│ COUCHE 2 — DEEP LEARNING (CRISP-DM Ph.4) │
│ CNN → LSTM → Dense │
│ Détection d'anomalies : Point | Contextuelle | Collective│
│ MLflow tracking : RMSE, MAE, R² │
└──────────────┬──────────────────────┬───────────────────┘
↓ Métriques ↓ Anomalies
┌──────────────────────┐ ┌───────────────────────────────┐
│ COUCHE 3 — FEEDBACK │ │ COUCHE 4 — RAG PIPELINE │
│ LOOP (ML↔Agents) │ │ Qdrant + text-embedding-3 │
│ │ │ CRAG confidence scoring │
│ FeedbackTrigger : │ │ DocumentReranker │
│ - RMSE > threshold │ │ ContextCompressor │
│ - N periods consec. │ └───────────────┬───────────────┘
│ - Manual trigger │ ↓
└──────────────┬───────┘ ┌───────────────────────────────┐
↓ │ COUCHE 5 — LLM AGENTS │
┌──────────────────────────────────────────────────────────┐
│ SupervisorGraph (LangGraph) │
│ │
│ AnalystAgent ResearcherAgent ReporterAgent │
│ - detect_anomaly_type - RAG + CRAG - submit_report│
│ - submit_feedback - Query reform. - LLM-as-judge │
│ - Reflexion pattern - Conf. scoring - Max 2 regen │
│ - SuggestionValidator │
│ │
│ Context Engineering (Zhang arXiv:2510.04618) │
│ Lost-in-the-middle : Critical→START | RAG→MID | Task→END│
└──────────────────────────┬───────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────┐
│ COUCHE 6 — SÉCURITÉ │
│ HMACGuard : sign_suggestion() | verify_suggestion() │
│ DataAnonymizer : PII removal avant envoi à Claude │
│ Prompt injection detection sur docs Qdrant │
│ GDPR compliance (Huang arXiv:2410.11182) │
└─────────────────────────┬───────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────┐
│ COUCHE 7 — BACKEND API (FastAPI) │
│ JWT Auth (3 rôles : admin|analyst|viewer) │
│ 8 routers : auth, health, predictions, anomalies, │
│ agents, feedback, reports │
│ PostgreSQL/TimescaleDB (asyncpg + threading pattern) │
│ uvloop compatible — thread-based async DB access │
└─────────────────────────┬───────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────┐
│ COUCHE 8 — DASHBOARD (Streamlit) │
│ 5 pages : Overview | Anomalies | Feedback HITL │
│ Reports | Agents | HITL History │
│ HITL : Approve/Reject avec raison obligatoire │
│ Rejection reasons → LongTermMemory (Mem0) │
│ Sidebar dynamique avec badges pending │
└─────────────────────────────────────────────────────────┘

---

## Stack technique

| Couche          | Technologies                                      |
| --------------- | ------------------------------------------------- |
| Deep Learning   | TensorFlow/Keras, CNN+LSTM, MLflow                |
| LLM Agents      | LangGraph, Anthropic Claude, Tool Use             |
| RAG             | Qdrant, OpenAI text-embedding-3-small, CRAG       |
| Mémoire         | ShortTermMemory (dict), LongTermMemory (Mem0)     |
| Sécurité        | HMAC-SHA-256, DataAnonymizer, Injection detection |
| Base de données | PostgreSQL + TimescaleDB, asyncpg                 |
| API             | FastAPI, JWT, uvicorn/uvloop                      |
| Dashboard       | Streamlit 1.32.0                                  |
| MLOps           | MLflow, Docker, GitHub Actions CI/CD              |

---

## Graphe LangGraph — SupervisorGraph

START
↓
supervisor_node
├── feedback_workflow
│ ├── FeedbackLoop.evaluate()
│ ├── AnalystAgent.analyze()
│ │ ├── detect_anomaly_type (Tool Use)
│ │ ├── submit_feedback_suggestion (Tool Use)
│ │ └── SuggestionValidator
│ └── HITL interrupt → human validates
│
├── analyse_workflow
│ ├── ResearcherAgent.research() — RAG + CRAG
│ └── AnalystAgent.analyze()
│
├── report_workflow
│ ├── ReporterAgent.generate_report()
│ └── LLM-as-a-judge (faithfulness + relevance + completeness)
│
└── research_workflow
└── ResearcherAgent.research()
END

---

## Flux HITL — Contribution originale PFE

CNN+LSTM résidus dégradés (RMSE > 0.15 × 5 périodes)
↓
FeedbackLoop.evaluate() déclenché
↓
DataAnonymizer.anonymize_metrics() — GDPR
↓
AnalystAgent.analyze() via Tool Use
↓
SuggestionValidator — bounds + confidence + justification
↓
HMACGuard.sign_suggestion() — intégrité garantie
↓
PostgreSQL INSERT (threading — uvloop compatible)
↓
Dashboard HITL — humain lit justification Claude
↓
Approve → UPDATE status=APPROVED → retraining schedulé
Reject → UPDATE status=REJECTED + reason → LongTermMemory
↓
Prochaine analyse : past_rejections récupérés
Claude ne repropose pas ce qui a été rejeté

---

## Différenciateurs vs état de l'art

| Système                             | Approche                              | Limite                     |
| ----------------------------------- | ------------------------------------- | -------------------------- |
| ARGOS (Microsoft, arXiv:2501.14170) | Règles statiques                      | Pas d'adaptation DL        |
| RAG classique                       | Retrieval sans évaluation             | Hallucinations             |
| MLOps classique                     | Retraining manuel                     | Lent, coûteux              |
| **Notre plateforme**                | **Boucle ML↔Agents bidirectionnelle** | **Contribution originale** |

---

## Papers académiques — justification de chaque composant

| Composant           | Paper                        |
| ------------------- | ---------------------------- |
| Architecture agents | Singh arXiv:2501.09136       |
| CRAG                | Yan arXiv:2401.15884         |
| Context Engineering | Zhang arXiv:2510.04618       |
| Tool Use fiabilité  | Dang arXiv:2509.18076        |
| detect_anomaly_type | Kang arXiv:2605.05725 (SAGE) |
| LLM-as-a-judge      | Aggarwal KDD 2025            |
| HMAC sécurité       | arXiv:2410.21492 (FATH)      |
| Data privacy        | Huang arXiv:2410.11182       |
| Mémoire agents      | Mem0 arXiv:2504.19413        |
| DL anomalies        | Darban arXiv:2211.05244      |
