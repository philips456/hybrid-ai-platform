# ══════════════════════════════════════════════════════
# Dockerfile — Hybrid AI Platform API
# Multi-stage build pour image légère en production
# ══════════════════════════════════════════════════════

# ── STAGE 1 : Builder ────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Dépendances système nécessaires pour compilation
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances Python
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# ── STAGE 2 : Production ─────────────────────────────
FROM python:3.12-slim AS production

WORKDIR /app

# Dépendances runtime uniquement
RUN apt-get update && apt-get install -y \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copier les packages installés depuis le builder
COPY --from=builder /root/.local /root/.local

# Copier le code source
COPY src/ ./src/
COPY configs/ ./configs/
COPY migrations/ ./migrations/

# Ajouter les packages locaux au PATH
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Utilisateur non-root pour la sécurité
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Port exposé
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Commande de démarrage
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4"]
