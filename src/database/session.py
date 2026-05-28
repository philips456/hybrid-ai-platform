"""
src/database/session.py
Gestion de la connexion PostgreSQL avec SQLAlchemy async.
Fournit get_db() pour l'injection de dépendances FastAPI.
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from configs.settings import settings

# ── MOTEUR ASYNC ──────────────────────────────────────
# asyncpg est le driver async pour PostgreSQL
DATABASE_URL = settings.database_url.replace(
    "postgresql://", "postgresql+asyncpg://"
)

engine = create_async_engine(
    DATABASE_URL,
    echo=False,           # True pour voir les requêtes SQL en dev
    pool_size=10,         # Connexions simultanées max
    max_overflow=20,      # Connexions supplémentaires si pool plein
    pool_pre_ping=True,   # Vérifie la connexion avant utilisation
)

# ── SESSION FACTORY ────────────────────────────────────
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Garde les objets accessibles après commit
)


# ── DEPENDENCY INJECTION FASTAPI ──────────────────────
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Générateur de session pour l'injection de dépendances FastAPI.

    Usage dans un router FastAPI :
        @router.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ── CONTEXT MANAGER POUR LES SCRIPTS ──────────────────
@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager pour utiliser la DB dans les scripts
    (hors FastAPI).

    Usage :
        async with get_db_context() as db:
            result = await db.execute(select(User))
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
