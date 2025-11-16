# MLOps Infrastructure Guide

This document describes the MLOps infrastructure available in this repository, both for local development and cloud deployment.

## 📋 Table of Contents

- [Local Development](#local-development)
- [Cloud Infrastructure (Terraform)](#cloud-infrastructure-terraform)
- [MLOps Practice Examples](#mlops-practice-examples)
- [Getting Started](#getting-started)

## 🏠 Local Development

### Docker Compose Services

The local environment includes MLOps-ready services:

| Service | Port | Purpose | URL |
|---------|------|---------|-----|
| **Jupyter Lab** | 8888 | Interactive ML development | http://localhost:8888 |
| **PostgreSQL** | 5432 | Experiment metadata storage | localhost:5432 |
| **MLflow** | 5000 | Experiment tracking & model registry | http://localhost:5000 |
| **PgAdmin** (optional) | 5050 | Database management UI | http://localhost:5050 |

### MLflow Tracking Server

MLflow is configured to use PostgreSQL as the backend store and local filesystem for artifact storage.

**Configuration:**
```bash
# Backend store: PostgreSQL
MLFLOW_BACKEND_STORE_URI=postgresql://mluser:mlpassword@postgres:5432/ml_practice

# Artifact storage: Local filesystem (mounted volume)
MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts

# Tracking URI from Jupyter
MLFLOW_TRACKING_URI=http://mlflow:5000
```

**Features:**
- ✅ Experiment tracking
- ✅ Model registry with versioning
- ✅ Model staging (None → Staging → Production → Archived)
- ✅ Artifact storage (models, plots, datasets)
- ✅ Metrics and parameters logging
- ✅ Run comparison
- ✅ Web UI for visualization

**Usage Example:**
```python
import mlflow
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("my_experiment")

with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.sklearn.log_model(model, "model")
```

### Quick Start Commands

```bash
# Start all MLOps services
make up

# View MLflow UI
open http://localhost:5000

# Access Jupyter for ML development
open http://localhost:8888

# View MLflow logs
make logs-mlflow

# Access MLflow container
make mlflow-shell
```

## ☁️ Cloud Infrastructure (Terraform)

### Architecture Overview

The Terraform configuration provisions a complete MLOps infrastructure on AWS:

```
┌─────────────────────────────────────────────────────────────┐
│                         AWS Cloud                            │
│                                                              │
│  ┌────────────┐    ┌──────────────┐    ┌─────────────┐   │
│  │   S3       │◄───│  SageMaker   │────►│ CloudWatch  │   │
│  │  Bucket    │    │  Notebook    │    │  Monitoring │   │
│  └────────────┘    └──────────────┘    └─────────────┘   │
│       │                                                      │
│       │            ┌──────────────┐                         │
│       ├───────────►│  SageMaker   │                         │
│       │            │   Training   │                         │
│       │            └──────────────┘                         │
│       │                                                      │
│       │            ┌──────────────┐    ┌─────────────┐   │
│       ├───────────►│  SageMaker   │────►│ Auto-       │   │
│       │            │  Endpoint    │    │ scaling     │   │
│       │            └──────────────┘    └─────────────┘   │
│       │                                                      │
│       │            ┌──────────────┐                         │
│       └───────────►│   Feature    │                         │
│                    │    Store     │                         │
│                    └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

### Modules

#### 1. S3 Module (`terraform/modules/s3/`)

Provisions S3 buckets for ML data and artifacts:

**Resources:**
- S3 bucket with versioning
- Server-side encryption (AES256 or KMS)
- Lifecycle rules for cost optimization
- Folder structure for ML workflows

**Folder Structure:**
```
s3://ml-practice-ml-{account-id}/
├── raw-data/
├── processed-data/
├── training-data/
├── validation-data/
├── test-data/
├── models/
├── model-artifacts/
├── feature-store/
├── data-capture/
├── experiments/
├── notebooks/
├── scripts/
└── temp/
```

**Lifecycle Rules:**
- Training data → Glacier after 90 days
- Models → Glacier IR after 180 days
- Temporary data → Deleted after 7 days
- Old versions → Deleted after 90 days

#### 2. SageMaker Module (`terraform/sagemaker/`)

Provisions SageMaker resources for ML workloads:

**Resources:**
- **Notebook Instance**: Interactive development environment
  - Configurable instance type (ml.t3.medium - ml.p3.2xlarge)
  - Lifecycle configuration for auto-setup
  - Git repository integration

- **Model Registry**: Centralized model versioning
  - Model package groups
  - Version tracking
  - Approval workflows

- **Endpoints**: Real-time inference
  - Auto-scaling support
  - Serverless inference option
  - Data capture for monitoring
  - CloudWatch metrics & alarms

- **Feature Store**: Feature management
  - Online store (low-latency)
  - Offline store (training)
  - Glue Data Catalog integration

- **IAM Roles**: Least-privilege access
  - SageMaker execution role
  - S3 access policies
  - ECR access for custom containers

**Monitoring:**
- CloudWatch log groups
- Endpoint invocation metrics
- Latency alarms
- Error rate alarms

**Cost Optimization:**
- Resources disabled by default
- Spot instance support
- Auto-scaling configuration
- Lifecycle policies

### Deployment

```bash
cd terraform/

# Initialize Terraform
terraform init

# Plan infrastructure (review before applying)
terraform plan

# Deploy infrastructure
terraform apply

# Outputs
terraform output
```

**Configuration Files:**
```bash
# Copy and customize
cp terraform.tfvars.example terraform.tfvars

# Edit variables
vim terraform.tfvars
```

**Key Variables:**
```hcl
# Enable/disable resources to control costs
create_sagemaker_notebook = false  # ~$50/month
create_sagemaker_endpoint = false  # ~$150/month
create_feature_store      = false  # Pay per request

# Instance types
sagemaker_notebook_instance_type = "ml.t3.medium"  # $0.05/hour
sagemaker_endpoint_instance_type = "ml.m5.large"   # $0.115/hour

# Auto-scaling
sagemaker_autoscaling_min_capacity = 1
sagemaker_autoscaling_max_capacity = 3
```

### Cost Estimates

| Resource | Configuration | Monthly Cost (us-east-1) |
|----------|--------------|--------------------------|
| S3 Bucket | 100GB + requests | ~$3 |
| SageMaker Notebook | ml.t3.medium, 8h/day | ~$12 |
| SageMaker Notebook | ml.t3.medium, 24/7 | ~$37 |
| SageMaker Endpoint | ml.m5.large, 1 instance | ~$83 |
| SageMaker Endpoint | ml.m5.large, auto-scaled 1-3 | ~$83-249 |
| Feature Store Online | 1M reads/month | ~$1.25 |
| Feature Store Offline | Storage only | Included in S3 |
| CloudWatch Logs | 5GB ingestion | ~$2.50 |
| **Minimal Setup** | S3 + Model Registry | **~$3/month** |
| **Dev Setup** | + Notebook (8h/day) | **~$15/month** |
| **Production** | + Endpoint + Monitoring | **~$90/month** |

## 🔧 MLOps Practice Examples

New MLOps examples in `code_folder/` (Category 19):

### Existing Examples (5)
1. `19_model_serving_fastapi.py` - REST API for model serving
2. `19_model_monitoring_prometheus.py` - Model performance monitoring
3. `19_data_drift_detection.py` - Data drift detection
4. `19_ab_testing_ml_models.py` - A/B testing framework
5. `19_feature_engineering_pipeline.py` - Feature engineering pipelines

### New Examples (3)
6. **`19_mlflow_model_registry.py`** - MLflow model versioning & lifecycle
   - Experiment tracking
   - Model registration
   - Stage transitions (Staging → Production)
   - Model loading and inference

7. **`19_bentoml_model_packaging.py`** - Model packaging & containerization
   - BentoML service creation
   - API generation
   - Docker containerization
   - Multi-cloud deployment

8. **`19_sagemaker_training_deployment.py`** - AWS SageMaker workflow
   - Training jobs
   - Model deployment
   - Endpoint creation
   - Inference at scale

9. **`19_great_expectations_data_validation.py`** - Data quality & validation
   - Expectation suites
   - Data validation
   - Quality reports
   - CI/CD integration

### Running Examples

```bash
# Run example in Jupyter container
make run-script SCRIPT=code_folder/19_mlflow_model_registry.py

# Or in Jupyter notebook
%run /home/jovyan/work/code_folder/19_mlflow_model_registry.py
```

## 🚀 Getting Started

### 1. Local Development Setup

```bash
# Clone repository
git clone <repository-url>
cd ml_problem_practice

# Setup environment
make setup

# Start MLOps services
make up

# Access services
# - Jupyter Lab: http://localhost:8888 (token: jupyter)
# - MLflow UI: http://localhost:5000
# - PostgreSQL: localhost:5432
```

### 2. Run MLOps Examples

```python
# In Jupyter notebook
import mlflow
import sys
sys.path.append('/home/jovyan/work/code_folder')

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://mlflow:5000")

# Run example
%run /home/jovyan/work/code_folder/19_mlflow_model_registry.py
```

### 3. Deploy Cloud Infrastructure (Optional)

```bash
# Configure AWS credentials
aws configure

# Navigate to terraform
cd terraform/

# Customize variables
cp terraform.tfvars.example terraform.tfvars
vim terraform.tfvars

# Deploy (start with minimal setup)
terraform init
terraform plan
terraform apply
```

### 4. Connect Local to Cloud

```python
# Use SageMaker from Jupyter
import sagemaker
import boto3

# Configure session
session = sagemaker.Session()
role = "arn:aws:iam::ACCOUNT_ID:role/sagemaker-execution-role"

# Upload data to S3
bucket = "ml-practice-ml-ACCOUNT_ID"
s3_input = session.upload_data(path='data/', bucket=bucket, key_prefix='training-data')

# Run training job
from sagemaker.sklearn import SKLearn

estimator = SKLearn(
    entry_point='train.py',
    role=role,
    instance_type='ml.m5.large',
    framework_version='1.0-1'
)

estimator.fit({'train': s3_input})
```

## 📚 Additional Resources

### Documentation
- [SETUP.md](SETUP.md) - Complete setup guide
- [terraform/README.md](terraform/README.md) - Terraform details
- [terraform/sagemaker/README.md](terraform/sagemaker/README.md) - SageMaker module docs

### Tools Documentation
- [MLflow](https://mlflow.org/docs/latest/index.html)
- [BentoML](https://docs.bentoml.org/)
- [Great Expectations](https://docs.greatexpectations.io/)
- [AWS SageMaker](https://docs.aws.amazon.com/sagemaker/)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)

### Example Workflows

**Experiment Tracking Workflow:**
```
1. Train models → Log to MLflow
2. Compare experiments → MLflow UI
3. Register best model → Model Registry
4. Transition to Staging → Test
5. Promote to Production → Deploy
```

**Cloud Training Workflow:**
```
1. Upload data to S3
2. Submit SageMaker training job
3. Monitor via CloudWatch
4. Register model in SageMaker Registry
5. Deploy to endpoint
6. Enable auto-scaling
7. Monitor with data capture
```

**CI/CD Workflow:**
```
1. Data validation → Great Expectations
2. Model training → MLflow tracking
3. Model packaging → BentoML
4. Containerization → Docker
5. Deployment → SageMaker/Kubernetes
6. Monitoring → CloudWatch/Prometheus
```

## 🛡️ Best Practices

### Local Development
- ✅ Use MLflow for all experiments
- ✅ Version datasets with DVC or S3 versioning
- ✅ Validate data with Great Expectations
- ✅ Package models with BentoML
- ✅ Back up PostgreSQL database regularly

### Cloud Deployment
- ✅ Start with minimal setup (S3 + Model Registry)
- ✅ Use Spot instances for training (up to 90% savings)
- ✅ Enable auto-scaling for endpoints
- ✅ Set up CloudWatch alarms
- ✅ Use tags for cost allocation
- ✅ Enable data capture for monitoring
- ✅ Store secrets in AWS Secrets Manager

### Security
- ✅ Never commit .env files
- ✅ Use IAM roles with least privilege
- ✅ Enable S3 encryption
- ✅ Block public S3 access
- ✅ Use VPC for production endpoints
- ✅ Enable CloudTrail logging
- ✅ Rotate credentials regularly

## 🔗 Quick Links

- [MLflow UI](http://localhost:5000)
- [Jupyter Lab](http://localhost:8888)
- [PostgreSQL](localhost:5432)
- [MLOps Examples](code_folder/)
- [Terraform Modules](terraform/)
