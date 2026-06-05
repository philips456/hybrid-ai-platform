src/
├── pages/
│ ├── Login.tsx ← Auth JWT
│ ├── Overview.tsx ← KPIs + graphiques temps réel
│ ├── Transactions.tsx ← Table des transactions
│ ├── Anomalies.tsx ← Alertes fraudes détectées
│ ├── HITL.tsx ← Validation humaine suggestions
│ ├── History.tsx ← Historique décisions
│ ├── Agents.tsx ← Run agents + résultats
│ ├── MLflow.tsx ← Experiments tracking
│ └── Settings.tsx ← Configuration
│
├── components/
│ ├── KPICard.tsx ← Métriques F1, AUC, RMSE
│ ├── TransactionChart.tsx ← Plotly/Recharts
│ ├── FraudMap.tsx ← Carte géographique
│ ├── SuggestionCard.tsx ← Carte suggestion HITL
│ ├── AgentPipeline.tsx ← Visualisation pipeline
│ └── Navbar.tsx ← Navigation + badges
│
├── api/
│ └── client.ts ← Appels FastAPI
│
└── types/
└── index.ts ← TypeScript interfaces
