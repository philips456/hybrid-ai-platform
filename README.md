# Hybrid AI Platform — PFE Nomo Philippe André
> Plateforme hybride IA : Deep Learning + LLM Agents + RAG + MLOps
> Méthodologie CRISP-DM — ESPRIT School of Engineering / Philipps-Universität Marburg

## Architecture globale
```
[Data Sources]
      │
      ▼
[CRISP-DM Phase 2-3] ── data_pipeline ──► processed/
      │
      ▼
[CRISP-DM Phase 4a] ── CNN+LSTM ──► Predictions + Anomalies
      │                                        │
      │          ◄──── Feedback Loop ──────────┘
      ▼
[CRISP-DM Phase 4b] ── LangGraph Agents ──► Reports
      │
      ▼
[CRISP-DM Phase 6] ── FastAPI ──► React Dashboard
```

## Démarrage rapide
```bash
cp configs/.env.example configs/.env   # Remplir les clés API
docker-compose up -d                   # Lance PostgreSQL, Qdrant, MLflow
python scripts/setup_db.py             # Crée les tables
python scripts/ingest_documents.py     # Indexe les documents dans Qdrant
uvicorn src.api.main:app --reload      # Lance l'API
```

## Stack technique
| Composant | Technologie |
|-----------|-------------|
| Deep Learning | TensorFlow/Keras + ONNX |
| Orchestration agents | LangGraph + LangChain |
| LLM | Anthropic Claude (claude-sonnet-4-20250514) |
| Base vectorielle | Qdrant (EU Frankfurt) |
| Base relationnelle | PostgreSQL + TimescaleDB |
| API | FastAPI + Pydantic v2 |
| MLOps | MLflow + Docker + GitHub Actions |
| Évaluation agents | LangSmith |
| Frontend | React + TypeScript + Tailwind CSS |
