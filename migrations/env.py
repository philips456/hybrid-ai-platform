"""
migrations/env.py
Configuration de l'environnement Alembic.
Charge automatiquement l'URL de la DB depuis settings.py
et détecte les changements de modèles (autogenerate).
"""
import asyncio
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# Ajouter le dossier racine au PATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.settings import settings
from src.database.base import Base
from src.database.models import (  # noqa — importer pour détection auto
    AgentRun, Alert, Anomaly, DataPoint, DataSource,
    FeedbackSuggestion, MLModel, Prediction, Report,
    TimeSeries, User,
)

# Configuration Alembic
config = context.config

# Surcharger l'URL avec celle de settings.py
config.set_main_option("sqlalchemy.url", settings.database_url)

# Logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Métadonnées pour l'autogenerate
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """
    Mode offline : génère les scripts SQL sans connexion DB.
    Utile pour générer des scripts à appliquer manuellement.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,      # Détecte les changements de type
        compare_server_default=True,  # Détecte les changements de default
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    Mode online async : applique les migrations directement.
    Utilisé pour les migrations PostgreSQL avec asyncpg.
    """
    database_url = settings.database_url.replace(
        "postgresql://", "postgresql+asyncpg://"
    )

    connectable = async_engine_from_config(
        {"sqlalchemy.url": database_url},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
