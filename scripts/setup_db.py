"""
scripts/setup_db.py
Crée toutes les tables PostgreSQL + TimescaleDB.
Lance les migrations Alembic et configure l'hypertable.

Usage:
    python scripts/setup_db.py
    python scripts/setup_db.py --reset
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from configs.settings import settings
from src.database.base import Base
from src.database.models import (  # noqa
    AgentRun, Alert, Anomaly, DataPoint, DataSource,
    FeedbackSuggestion, MLModel, Prediction, Report,
    TimeSeries, User,
)


async def create_extensions(engine):
    print("📦 Activation des extensions PostgreSQL...")
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))
        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";'))
    print("   ✅ TimescaleDB + UUID activés")


async def create_tables(engine):
    print("🗄️  Création des tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("   ✅ 11 tables créées")


async def create_hypertable(engine):
    print("⚡ Configuration de l'hypertable TimescaleDB...")
    async with engine.begin() as conn:
        result = await conn.execute(text("""
            SELECT COUNT(*) FROM timescaledb_information.hypertables
            WHERE hypertable_name = 'data_points'
        """))
        count = result.scalar()
        if count == 0:
            await conn.execute(text("""
                SELECT create_hypertable(
                    'data_points', 'timestamp',
                    chunk_time_interval => INTERVAL '1 month',
                    if_not_exists => TRUE
                );
            """))
            print("   ✅ data_points convertie en hypertable (partition mensuelle)")
        else:
            print("   ℹ️  data_points déjà configurée en hypertable")


async def create_indexes(engine):
    print("📇 Création des index de performance...")
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_predictions_series_time ON predictions (series_id, timestamp DESC);",
        "CREATE INDEX IF NOT EXISTS idx_anomalies_status_time ON anomalies (status, timestamp DESC);",
        "CREATE INDEX IF NOT EXISTS idx_alerts_unprocessed ON alerts (is_processed, created_at DESC) WHERE is_processed = FALSE;",
        "CREATE INDEX IF NOT EXISTS idx_agent_runs_type_time ON agent_runs (agent_type, created_at DESC);",
        "CREATE INDEX IF NOT EXISTS idx_feedback_pending ON feedback_suggestions (status, created_at DESC) WHERE status = 'PENDING'::feedbackstatus;",
    ]
    async with engine.begin() as conn:
        for idx_sql in indexes:
            await conn.execute(text(idx_sql))
    print(f"   ✅ {len(indexes)} index créés")


async def insert_default_admin(engine):
    print("👤 Création de l'utilisateur admin...")
    import bcrypt
    async with engine.begin() as conn:
        result = await conn.execute(text("SELECT COUNT(*) FROM users WHERE username = 'admin'"))
        if result.scalar() == 0:
            hashed = bcrypt.hashpw("admin123".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
            await conn.execute(text("""
                INSERT INTO users (id, username, email, hashed_password, role, is_active)
                VALUES (uuid_generate_v4(), 'admin', 'admin@hybrid-ai-platform.local', :pwd, 'ADMIN', TRUE)
            """), {"pwd": hashed})
            print("   ✅ Admin créé (user: admin / pass: admin123)")
            print("   ⚠️  Changer le mot de passe en production !")
        else:
            print("   ℹ️  Admin déjà existant")


async def insert_synthetic_source(engine):
    print("🧪 Création de la source synthétique...")
    async with engine.begin() as conn:
        result = await conn.execute(text("SELECT COUNT(*) FROM data_sources WHERE name = 'synthetic_test'"))
        if result.scalar() == 0:
            await conn.execute(text("""
                INSERT INTO data_sources (id, name, domain, source_type, is_active)
                VALUES (uuid_generate_v4(), 'synthetic_test', 'SYNTHETIC', 'synthetic', TRUE)
            """))
            print("   ✅ Source synthétique créée")
        else:
            print("   ℹ️  Source synthétique déjà existante")


async def drop_all_tables(engine):
    print("🗑️  Suppression de toutes les tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    print("   ✅ Tables supprimées")


async def verify_setup(engine):
    print("\n🔍 Vérification de la configuration...")
    async with engine.connect() as conn:
        result = await conn.execute(text("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        """))
        table_count = result.scalar()
        result = await conn.execute(text("""
            SELECT COUNT(*) FROM timescaledb_information.hypertables
            WHERE hypertable_name = 'data_points'
        """))
        is_hypertable = result.scalar() > 0
        result = await conn.execute(text("SELECT COUNT(*) FROM users"))
        user_count = result.scalar()

    print(f"\n{'='*50}")
    print(f"  Tables créées    : {table_count}/11")
    print(f"  Hypertable       : {'✅ Oui' if is_hypertable else '❌ Non'}")
    print(f"  Utilisateurs     : {user_count}")
    print(f"{'='*50}")
    print("  🎉 Base de données prête !")
    print(f"{'='*50}\n")


async def main(reset=False):
    print("\n" + "="*50)
    print("  Hybrid AI Platform — Setup Base de données")
    print("="*50 + "\n")

    database_url = settings.database_url.replace(
        "postgresql://", "postgresql+asyncpg://"
    )
    engine = create_async_engine(database_url, echo=False)

    try:
        if reset:
            await drop_all_tables(engine)
        await create_extensions(engine)
        await create_tables(engine)
        await create_hypertable(engine)
        await create_indexes(engine)
        await insert_default_admin(engine)
        await insert_synthetic_source(engine)
        await verify_setup(engine)
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        print("Vérifiez que PostgreSQL est démarré : docker-compose up -d postgres")
        sys.exit(1)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(reset=args.reset))
