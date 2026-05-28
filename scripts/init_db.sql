-- ══════════════════════════════════════════════════════
-- init_db.sql — Initialisation PostgreSQL + TimescaleDB
-- Exécuté automatiquement au premier démarrage Docker
-- ══════════════════════════════════════════════════════

-- Activer TimescaleDB
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Activer UUID
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Message de confirmation
SELECT 'Extensions TimescaleDB et UUID activées avec succès' AS status;
