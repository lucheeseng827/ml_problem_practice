# ML Practice Development Environment Setup

This guide will help you set up a complete local development environment for ML practice using Docker Compose.

## 🏗️ Architecture Overview

The development environment includes:

- **Jupyter Lab**: Interactive notebook environment with all ML libraries pre-installed
- **PostgreSQL**: Relational database for storing experiment metadata and structured data
- **DuckDB**: Embedded analytical database for fast data analysis
- **PgAdmin** (optional): Web-based database management tool

## 📋 Prerequisites

- Docker Engine 20.10+ and Docker Compose V2
- At least 8GB RAM available for Docker
- At least 20GB free disk space
- Git

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ml_problem_practice
```

### 2. Set Up Environment Variables

Copy the example environment file and customize it:

```bash
cp .env.example .env
```

Edit `.env` file if you want to change default passwords or configurations.

### 3. Start the Development Environment

```bash
# Start all services
docker-compose up -d

# Or start without PgAdmin (lighter setup)
docker-compose up -d jupyter postgres

# View logs
docker-compose logs -f
```

### 4. Access Services

Once containers are running:

- **Jupyter Lab**: http://localhost:8888 (token: `jupyter` by default)
- **PostgreSQL**: `localhost:5432`
  - Database: `ml_practice`
  - User: `mluser`
  - Password: `mlpassword`
- **PgAdmin** (if started): http://localhost:5050
  - Email: `admin@mlpractice.local`
  - Password: `admin`

## 📁 Directory Structure

```
ml_problem_practice/
├── code_folder/           # ML practice scripts (Categories 1-20)
├── notebooks/            # Jupyter notebooks (auto-created)
├── data/                 # Datasets (auto-created)
├── models/               # Saved models (auto-created)
├── duckdb/              # DuckDB database files (auto-created)
├── sql/init/            # PostgreSQL initialization scripts
├── docker/              # Docker configuration files
├── terraform/           # Terraform configurations (for later)
├── docker-compose.yml   # Docker Compose configuration
├── requirements.txt     # Python dependencies
└── .env                 # Environment variables (create from .env.example)
```

## 🔧 Using the Environment

### Working with Jupyter Lab

1. Open Jupyter Lab at http://localhost:8888
2. Navigate to `work/code_folder/` to access ML practice scripts
3. Create new notebooks in `work/notebooks/`
4. Access data in `work/data/`

Example notebook to test the setup:

```python
# Test imports
import pandas as pd
import numpy as np
import duckdb
import psycopg2
from sqlalchemy import create_engine

# Test DuckDB
conn = duckdb.connect('/home/jovyan/work/duckdb/ml_practice.db')
conn.execute("SELECT 'DuckDB is working!' as message").fetchall()

# Test PostgreSQL connection
db_url = "postgresql://mluser:mlpassword@postgres:5432/ml_practice"
engine = create_engine(db_url)
pd.read_sql("SELECT 'PostgreSQL is working!' as message", engine)
```

### Working with PostgreSQL

#### From Jupyter:

```python
import psycopg2
from sqlalchemy import create_engine
import pandas as pd

# Using SQLAlchemy (recommended)
engine = create_engine('postgresql://mluser:mlpassword@postgres:5432/ml_practice')

# Save experiment data
df = pd.DataFrame({
    'experiment_name': ['test_model_1'],
    'model_type': ['random_forest'],
    'dataset_name': ['iris']
})
df.to_sql('experiments', engine, schema='ml_experiments', if_exists='append', index=False)

# Query experiments
pd.read_sql("SELECT * FROM ml_experiments.experiment_summary", engine)
```

#### From Host Machine:

```bash
# Connect using psql
docker exec -it ml-practice-postgres psql -U mluser -d ml_practice

# Or use any PostgreSQL client
# Host: localhost
# Port: 5432
# Database: ml_practice
# User: mluser
# Password: mlpassword
```

### Working with DuckDB

DuckDB is excellent for analytical queries on large datasets:

```python
import duckdb

# Connect to DuckDB (file-based)
conn = duckdb.connect('/home/jovyan/work/duckdb/ml_practice.db')

# Load CSV directly
conn.execute("""
    CREATE TABLE IF NOT EXISTS iris AS
    SELECT * FROM read_csv_auto('/home/jovyan/work/data/iris.csv')
""")

# Query data
conn.execute("SELECT * FROM iris LIMIT 5").df()

# DuckDB can also query Pandas DataFrames directly
import pandas as pd
df = pd.read_csv('/home/jovyan/work/data/iris.csv')
conn.execute("SELECT * FROM df WHERE sepal_length > 5.0").df()
```

## 🛠️ Common Operations

### Stop Services

```bash
# Stop all containers
docker-compose down

# Stop and remove volumes (WARNING: deletes all data)
docker-compose down -v
```

### Restart Services

```bash
docker-compose restart
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f jupyter
docker-compose logs -f postgres
```

### Rebuild After Changing Dependencies

```bash
# Rebuild Jupyter container
docker-compose build jupyter

# Restart with new build
docker-compose up -d --build jupyter
```

### Install Additional Python Packages

**Option 1**: Add to `requirements.txt` and rebuild

```bash
echo "your-package>=1.0.0" >> requirements.txt
docker-compose build jupyter
docker-compose up -d jupyter
```

**Option 2**: Install temporarily (lost on container restart)

```python
# In Jupyter notebook
!pip install your-package
```

### Backup Database

```bash
# Backup PostgreSQL
docker exec ml-practice-postgres pg_dump -U mluser ml_practice > backup.sql

# Restore PostgreSQL
docker exec -i ml-practice-postgres psql -U mluser ml_practice < backup.sql

# Backup DuckDB (just copy the file)
cp duckdb/ml_practice.db duckdb/ml_practice_backup.db
```

## 📊 Database Schema

The PostgreSQL database comes pre-initialized with the following schema:

### Schema: `ml_experiments`

**Tables:**
- `experiments`: Track ML experiments
- `metrics`: Store model metrics (accuracy, loss, etc.)
- `hyperparameters`: Store hyperparameter configurations
- `datasets`: Metadata about datasets
- `artifacts`: Track model artifacts and files

**Views:**
- `experiment_summary`: Summary of all experiments

See `sql/init/01_init.sql` for complete schema.

## 🎯 Running ML Practice Examples

The `code_folder/` contains 48+ ML practice examples across 20 categories:

```bash
# Enter Jupyter container
docker exec -it ml-practice-jupyter bash

# Run a practice script
cd /home/jovyan/work/code_folder
python 10_statsmodels_arima_sarima.py
```

Or run from a Jupyter notebook:

```python
# In Jupyter
%run /home/jovyan/work/code_folder/10_statsmodels_arima_sarima.py
```

## 🔍 Troubleshooting

### Jupyter Token Issues

If you can't access Jupyter, get the token:

```bash
docker exec ml-practice-jupyter jupyter server list
```

### Port Already in Use

If ports 8888, 5432, or 5050 are already in use, edit `docker-compose.yml`:

```yaml
ports:
  - "8889:8888"  # Change 8889 to any available port
```

### Container Won't Start

Check logs:

```bash
docker-compose logs jupyter
docker-compose logs postgres
```

### Database Connection Issues

Ensure PostgreSQL is healthy:

```bash
docker-compose ps
docker exec ml-practice-postgres pg_isready -U mluser
```

### Memory Issues

Increase Docker memory allocation in Docker Desktop settings (minimum 8GB recommended).

## 🚀 Next Steps: Cloud Infrastructure (Terraform)

After setting up the local environment, you can provision cloud resources using Terraform:

```bash
cd terraform/

# Initialize Terraform
terraform init

# Plan infrastructure
terraform plan

# Apply infrastructure
terraform apply
```

**Terraform will provision:**
- AWS SageMaker for model training
- AWS Athena for SQL queries on S3
- AWS Glue for ETL jobs
- AWS Redshift for data warehousing

See `terraform/README.md` for detailed instructions (coming soon).

## 📚 Additional Resources

- [Jupyter Lab Documentation](https://jupyterlab.readthedocs.io/)
- [DuckDB Documentation](https://duckdb.org/docs/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)

## 🤝 Contributing

When adding new ML practice examples:

1. Add the script to `code_folder/`
2. Update `code_folder/README_NEW_CATEGORIES.md`
3. If new dependencies are needed, add to `requirements.txt`
4. Test in the Docker environment before committing

## 📝 License

See LICENSE file for details.
