"""
src/database/base.py
Classe de base commune à tous les modèles SQLAlchemy
Fournit : id UUID, created_at, updated_at automatiques
"""
import uuid
from datetime import datetime

from sqlalchemy import DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Classe de base pour tous les modèles SQLAlchemy."""
    pass


class TimestampMixin:
    """
    Mixin qui ajoute created_at et updated_at automatiques.
    Hérité par tous les modèles qui ont besoin de traçabilité temporelle.
    """
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )


class UUIDMixin:
    """
    Mixin qui ajoute une clé primaire UUID v4.
    Utilisé par tous les modèles pour garantir l'unicité
    à l'échelle distribuée.
    """
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False
    )
