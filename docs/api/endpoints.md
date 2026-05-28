# API Endpoints Documentation

## Authentification
POST /auth/token — Obtenir un JWT

## Prédictions
POST /predict — Prédiction temps réel
GET  /predictions — Historique des prédictions

## Anomalies
GET  /anomalies — Liste des anomalies détectées
GET  /anomalies/{id} — Détail d'une anomalie

## Agents
POST /agents/run — Déclencher un run d'agent
GET  /agents/runs — Historique des runs

## Rapports
GET  /reports — Liste des rapports générés
GET  /reports/{id} — Récupérer un rapport

## Feedback Loop
POST /feedback/validate — Valider/rejeter une suggestion
GET  /feedback/pending — Suggestions en attente de validation

## Santé
GET  /health — Health check
GET  /ready — Readiness check
