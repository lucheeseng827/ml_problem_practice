# =====================================
# Pre-commit Hooks
# =====================================
precommit-all:
	pre-commit run --all-files

precommit-init:
	pre-commit install

precommit-update:
	pre-commit autoupdate

precommit-install:
	pip install pre-commit ./precommit_requirements.txt

# =====================================
# Docker Development Environment
# =====================================

# Setup: Create .env from example
setup:
	@echo "Setting up development environment..."
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✓ Created .env file from .env.example"; \
		echo "  Please edit .env if you need to customize settings"; \
	else \
		echo "✓ .env file already exists"; \
	fi
	@echo "✓ Setup complete!"

# Start all services
up:
	@echo "Starting all services..."
	docker-compose up -d
	@echo "✓ Services started!"
	@echo ""
	@echo "Access points:"
	@echo "  - Jupyter Lab: http://localhost:8888 (token: jupyter)"
	@echo "  - PostgreSQL: localhost:5432 (user: mluser, password: mlpassword)"
	@echo "  - MLflow: http://localhost:5000"
	@echo ""
	@echo "Run 'make logs' to view logs"

# Start services with PgAdmin
up-full:
	@echo "Starting all services including PgAdmin..."
	docker-compose --profile tools up -d
	@echo "✓ Services started!"
	@echo ""
	@echo "Access points:"
	@echo "  - Jupyter Lab: http://localhost:8888 (token: jupyter)"
	@echo "  - PostgreSQL: localhost:5432 (user: mluser, password: mlpassword)"
	@echo "  - MLflow: http://localhost:5000"
	@echo "  - PgAdmin: http://localhost:5050 (email: admin@mlpractice.local, password: admin)"

# Stop all services
down:
	@echo "Stopping all services..."
	docker-compose down
	@echo "✓ Services stopped"

# Stop and remove volumes (WARNING: deletes all data)
down-volumes:
	@echo "⚠️  WARNING: This will delete all data in Docker volumes!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker-compose down -v; \
		echo "✓ Services stopped and volumes removed"; \
	else \
		echo "Cancelled"; \
	fi

# Restart services
restart:
	@echo "Restarting services..."
	docker-compose restart
	@echo "✓ Services restarted"

# View logs
logs:
	docker-compose logs -f

# View logs for specific service
logs-jupyter:
	docker-compose logs -f jupyter

logs-postgres:
	docker-compose logs -f postgres

logs-mlflow:
	docker-compose logs -f mlflow

# Build/rebuild containers
build:
	@echo "Building containers..."
	docker-compose build
	@echo "✓ Build complete"

# Build and start
build-up:
	@echo "Building and starting services..."
	docker-compose up -d --build
	@echo "✓ Services built and started"

# Check service status
status:
	docker-compose ps

# Access Jupyter container shell
jupyter-shell:
	docker exec -it ml-practice-jupyter bash

# Access PostgreSQL shell
postgres-shell:
	docker exec -it ml-practice-postgres psql -U mluser -d ml_practice

# Access MLflow container shell
mlflow-shell:
	docker exec -it ml-practice-mlflow sh

# Backup PostgreSQL database
backup-db:
	@echo "Backing up PostgreSQL database..."
	@mkdir -p backups
	docker exec ml-practice-postgres pg_dump -U mluser ml_practice > backups/ml_practice_$$(date +%Y%m%d_%H%M%S).sql
	@echo "✓ Backup created in backups/"

# Restore PostgreSQL database (usage: make restore-db FILE=backup.sql)
restore-db:
	@if [ -z "$(FILE)" ]; then \
		echo "Error: Please specify FILE=backup.sql"; \
		exit 1; \
	fi
	@echo "Restoring database from $(FILE)..."
	docker exec -i ml-practice-postgres psql -U mluser ml_practice < $(FILE)
	@echo "✓ Database restored"

# Clean up unused Docker resources
clean:
	@echo "Cleaning up unused Docker resources..."
	docker system prune -f
	@echo "✓ Cleanup complete"

# Run a Python script in Jupyter container
run-script:
	@if [ -z "$(SCRIPT)" ]; then \
		echo "Usage: make run-script SCRIPT=code_folder/script.py"; \
		exit 1; \
	fi
	docker exec ml-practice-jupyter python /home/jovyan/work/$(SCRIPT)

# Install additional Python package
install-package:
	@if [ -z "$(PACKAGE)" ]; then \
		echo "Usage: make install-package PACKAGE=package-name"; \
		exit 1; \
	fi
	@echo "Installing $(PACKAGE)..."
	docker exec ml-practice-jupyter pip install $(PACKAGE)
	@echo "✓ $(PACKAGE) installed"
	@echo "Note: Add to requirements.txt and rebuild for permanent installation"

# =====================================
# Help
# =====================================
help:
	@echo "ML Practice Development Environment - Available Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          - Initial setup (creates .env file)"
	@echo ""
	@echo "Docker Services:"
	@echo "  make up             - Start Jupyter + PostgreSQL + MLflow"
	@echo "  make up-full        - Start all services including PgAdmin"
	@echo "  make down           - Stop all services"
	@echo "  make down-volumes   - Stop services and remove volumes (deletes data)"
	@echo "  make restart        - Restart all services"
	@echo "  make build          - Build/rebuild containers"
	@echo "  make build-up       - Build and start services"
	@echo "  make status         - Show service status"
	@echo ""
	@echo "Logs:"
	@echo "  make logs           - View logs for all services"
	@echo "  make logs-jupyter   - View Jupyter logs"
	@echo "  make logs-postgres  - View PostgreSQL logs"
	@echo "  make logs-mlflow    - View MLflow logs"
	@echo ""
	@echo "Shell Access:"
	@echo "  make jupyter-shell  - Access Jupyter container shell"
	@echo "  make postgres-shell - Access PostgreSQL shell"
	@echo "  make mlflow-shell   - Access MLflow container shell"
	@echo ""
	@echo "Database Operations:"
	@echo "  make backup-db      - Backup PostgreSQL database"
	@echo "  make restore-db FILE=backup.sql - Restore database"
	@echo ""
	@echo "Utilities:"
	@echo "  make run-script SCRIPT=path/to/script.py - Run Python script"
	@echo "  make install-package PACKAGE=name - Install Python package"
	@echo "  make clean          - Clean up unused Docker resources"
	@echo ""
	@echo "Pre-commit Hooks:"
	@echo "  make precommit-init    - Install pre-commit hooks"
	@echo "  make precommit-all     - Run pre-commit on all files"
	@echo "  make precommit-update  - Update pre-commit hooks"
	@echo ""

.PHONY: setup up up-full down down-volumes restart logs logs-jupyter logs-postgres logs-mlflow \
        build build-up status jupyter-shell postgres-shell mlflow-shell backup-db restore-db \
        clean run-script install-package help \
        precommit-all precommit-init precommit-update precommit-install

# Path: Makefile
