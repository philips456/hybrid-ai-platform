# ══════════════════════════════════════════════════════
# Makefile — Commandes de développement
# Usage : make <commande>
# ══════════════════════════════════════════════════════

.PHONY: help install install-dev up down up-dev down-dev \
        test test-unit test-integration lint format \
        train migrate db-reset clean

# ── AIDE ─────────────────────────────────────────────
help:
	@echo "Commandes disponibles :"
	@echo ""
	@echo "  Setup"
	@echo "    make install       Installer les dépendances production"
	@echo "    make install-dev   Installer les dépendances développement"
	@echo "    make migrate       Lancer les migrations Alembic"
	@echo ""
	@echo "  Docker"
	@echo "    make up            Lancer tous les services (prod)"
	@echo "    make down          Arrêter tous les services"
	@echo "    make up-dev        Lancer en mode développement (hot reload)"
	@echo "    make down-dev      Arrêter le mode développement"
	@echo ""
	@echo "  Développement"
	@echo "    make test          Lancer tous les tests"
	@echo "    make test-unit     Tests unitaires uniquement"
	@echo "    make lint          Vérifier le code (flake8 + bandit)"
	@echo "    make format        Formater le code (black + isort)"
	@echo "    make train         Lancer l'entraînement du modèle DL"
	@echo ""
	@echo "  Maintenance"
	@echo "    make db-reset      Réinitialiser la base de données (dev)"
	@echo "    make clean         Nettoyer les fichiers temporaires"

# ── INSTALLATION ─────────────────────────────────────
install:
	pip install --upgrade pip
	pip install -r requirements.txt

install-dev:
	pip install --upgrade pip
	pip install -r requirements-dev.txt

# ── DOCKER ───────────────────────────────────────────
up:
	docker-compose up -d
	@echo "Services démarrés :"
	@echo "  API     → http://localhost:8000/docs"
	@echo "  MLflow  → http://localhost:5000"
	@echo "  Qdrant  → http://localhost:6333/dashboard"

down:
	docker-compose down

up-dev:
	docker-compose -f docker-compose.dev.yml up -d
	@echo "Mode développement démarré (hot reload activé)"

down-dev:
	docker-compose -f docker-compose.dev.yml down

# ── BASE DE DONNÉES ───────────────────────────────────
migrate:
	alembic upgrade head

db-reset:
	docker-compose -f docker-compose.dev.yml down -v
	docker-compose -f docker-compose.dev.yml up -d postgres
	sleep 5
	alembic upgrade head
	python scripts/setup_db.py
	@echo "Base de données réinitialisée"

# ── TESTS ─────────────────────────────────────────────
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v --cov=src

test-integration:
	pytest tests/integration/ -v

# ── QUALITÉ DU CODE ───────────────────────────────────
lint:
	flake8 src/ tests/ --max-line-length=100 --exclude=__pycache__
	bandit -r src/ -ll
	@echo "Lint terminé"

format:
	black src/ tests/ scripts/ configs/ --line-length=100
	isort src/ tests/ scripts/ configs/
	@echo "Formatage terminé"

# ── ENTRAÎNEMENT ─────────────────────────────────────
train:
	python scripts/train_model.py --config configs/model_config.yaml

# ── NETTOYAGE ─────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	rm -rf htmlcov/ 2>/dev/null || true
	@echo "Nettoyage terminé"
