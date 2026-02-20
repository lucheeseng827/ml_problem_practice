# Complete ML Model Lifecycle with Pre-trained Models and Pipeline Orchestration

This guide explains the complete lifecycle of ML models from development to production, with focus on using pre-trained models and orchestration frameworks.

## 📋 Table of Contents

- [Model Lifecycle Overview](#model-lifecycle-overview)
- [Using Pre-trained Models](#using-pre-trained-models)
- [Pipeline Orchestration](#pipeline-orchestration)
- [Scheduling Frameworks](#scheduling-frameworks)
- [Complete Examples](#complete-examples)
- [Production Validation & Argo Rollouts](#-production-validation--argo-rollouts)

## 🔄 Model Lifecycle Overview

### The Complete ML Model Lifecycle

```
┌─────────────────────────────────────────────────────────────────┐
│                     ML MODEL LIFECYCLE                           │
└─────────────────────────────────────────────────────────────────┘

1. PROBLEM DEFINITION
   ├── Define business objective
   ├── Identify success metrics
   └── Determine model requirements

2. DATA PIPELINE
   ├── Data Ingestion ───────► Extract from sources (DB, API, S3)
   ├── Data Validation ──────► Quality checks (Great Expectations)
   ├── Data Preprocessing ───► Clean, transform, feature engineering
   └── Data Versioning ──────► Track with DVC/S3 versioning

3. MODEL DEVELOPMENT
   ├── Model Selection ──────► Choose architecture (pre-trained vs scratch)
   │   ├── Pre-trained: BERT, ResNet, CLIP (Transfer Learning)
   │   └── Custom: Train from scratch if needed
   ├── Model Training ───────► Train with hyperparameter tuning
   │   ├── Feature Extraction (freeze base model)
   │   ├── Fine-tuning (unfreeze all layers)
   │   └── Gradual unfreezing (progressive training)
   ├── Model Evaluation ─────► Test metrics, validation
   └── Experiment Tracking ──► MLflow, Weights & Biases

4. MODEL VALIDATION
   ├── Quality Gates ────────► Minimum accuracy/F1 thresholds
   ├── Bias/Fairness Tests ──► Check for dataset bias
   ├── Performance Tests ────► Latency, throughput
   └── A/B Test Design ──────► Comparison strategy

5. MODEL DEPLOYMENT
   ├── Model Registration ───► MLflow Registry, SageMaker Registry
   │   └── Versioning: v1.0.0, v1.0.1, v2.0.0
   ├── Model Packaging ──────► BentoML, Docker, ONNX
   ├── Deployment ───────────► SageMaker Endpoint, Kubernetes, Lambda
   │   ├── Canary: 5% traffic to new model
   │   ├── Blue-Green: Switch traffic instantly
   │   └── Shadow: Run both, compare results
   └── Endpoint Configuration ► Auto-scaling, health checks

6. MONITORING & MAINTENANCE
   ├── Performance Monitoring ► Track accuracy, latency
   ├── Data Drift Detection ─► Input distribution changes
   ├── Concept Drift Detection► Target distribution changes
   ├── Alerting ─────────────► Notify on degradation
   └── Retraining Triggers ───► Automated based on drift

7. CONTINUOUS IMPROVEMENT
   ├── Collect Production Data ► New samples for retraining
   ├── Active Learning ───────► Query labeling for hard cases
   ├── Model Updates ────────► Retrain with new data
   └── Feedback Loop ────────► User feedback integration

8. GOVERNANCE & COMPLIANCE
   ├── Model Documentation ───► Model cards, datasheets
   ├── Audit Trail ──────────► Track all model versions
   ├── Explainability ───────► SHAP, LIME for predictions
   └── Regulatory Compliance ► GDPR, bias testing
```

## 🎯 Using Pre-trained Models

### Why Pre-trained Models?

**Benefits:**
- ✅ **Faster Development**: Skip weeks/months of training
- ✅ **Better Performance**: Leverage learning from massive datasets
- ✅ **Less Data Needed**: Transfer learning requires less training data
- ✅ **Proven Architectures**: Battle-tested models
- ✅ **Cost Effective**: Reduce GPU hours significantly

### Transfer Learning Strategies

#### 1. Feature Extraction (Freeze Base Model)

```python
# Example: BERT for text classification
from transformers import AutoModelForSequenceClassification

# Load pre-trained BERT
model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=3
)

# Freeze all base BERT parameters
for param in model.bert.parameters():
    param.requires_grad = False

# Only classification head is trainable
# Training time: ~1-2 hours instead of days
```

**Use when:**
- Limited training data
- Limited computational resources
- Task is similar to pre-training task
- Need fast iteration

#### 2. Fine-tuning (Unfreeze All Layers)

```python
# Unfreeze all parameters
for param in model.parameters():
    param.requires_grad = True

# All weights will be updated
# Training time: ~4-8 hours, better task adaptation
```

**Use when:**
- Sufficient training data (1000+ examples)
- Task differs from pre-training
- Need best possible performance
- Have computational resources

#### 3. Gradual Unfreezing (Progressive Training)

```python
# Stage 1: Train head only (2 epochs)
for param in model.bert.parameters():
    param.requires_grad = False
# ... train ...

# Stage 2: Unfreeze last 2 layers (2 epochs)
for param in model.bert.encoder.layer[-2:].parameters():
    param.requires_grad = True
# ... train ...

# Stage 3: Unfreeze all (2 epochs)
for param in model.parameters():
    param.requires_grad = True
# ... train ...
```

**Use when:**
- Moderate training data
- Want balance between speed and performance
- Prevent catastrophic forgetting

#### 4. Discriminative Learning Rates

```python
# Different learning rates for different layers
optimizer = torch.optim.Adam([
    {'params': model.bert.embeddings.parameters(), 'lr': 1e-5},
    {'params': model.bert.encoder.parameters(), 'lr': 2e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-4}
])

# Lower layers (more general) learn slower
# Upper layers (more task-specific) learn faster
```

### Popular Pre-trained Models

#### NLP (Text)
```python
# HuggingFace Transformers
models = {
    'BERT': 'bert-base-uncased',           # General purpose
    'RoBERTa': 'roberta-base',             # Better than BERT
    'DistilBERT': 'distilbert-base-uncased', # Faster, smaller
    'DeBERTa': 'microsoft/deberta-v3-base', # State-of-the-art
    'ELECTRA': 'google/electra-base-discriminator',
    'T5': 't5-base',                       # Text-to-text
    'GPT-2': 'gpt2',                       # Text generation
}
```

#### Computer Vision (Images)
```python
# PyTorch Vision
from torchvision import models

models_cv = {
    'ResNet': models.resnet50(weights='IMAGENET1K_V2'),
    'EfficientNet': models.efficientnet_b0(weights='IMAGENET1K_V1'),
    'Vision Transformer': models.vit_b_16(weights='IMAGENET1K_V1'),
    'Swin Transformer': models.swin_b(weights='IMAGENET1K_V1'),
    'ConvNeXt': models.convnext_base(weights='IMAGENET1K_V1'),
}
```

#### Multi-modal (Text + Images)
```python
models_multimodal = {
    'CLIP': 'openai/clip-vit-base-patch32',
    'BLIP': 'Salesforce/blip-image-captioning-base',
    'LayoutLM': 'microsoft/layoutlm-base-uncased',
}
```

### Model Adaptation Workflow

```python
# 1. Load pre-trained model
from transformers import AutoModel

base_model = AutoModel.from_pretrained('bert-base-uncased')

# 2. Add task-specific head
class CustomModel(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base = base_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids, attention_mask)
        pooled = outputs.pooler_output
        dropped = self.dropout(pooled)
        return self.classifier(dropped)

model = CustomModel(base_model, num_classes=3)

# 3. Configure training
# 4. Train on custom data
# 5. Evaluate
# 6. Deploy
```

## 🔧 Pipeline Orchestration

### Why Orchestration?

**Challenges without orchestration:**
- ❌ Manual execution of each step
- ❌ No dependency management
- ❌ Difficult to monitor
- ❌ Hard to debug failures
- ❌ No automatic retries
- ❌ Can't schedule recurring runs

**Benefits with orchestration:**
- ✅ **Automated Execution**: Entire pipeline runs automatically
- ✅ **Dependency Management**: Tasks run in correct order
- ✅ **Monitoring**: Track each task's status
- ✅ **Error Handling**: Automatic retries, alerts
- ✅ **Scheduling**: Daily, weekly, on-demand runs
- ✅ **Scalability**: Parallel task execution
- ✅ **Reproducibility**: Same pipeline, consistent results

### Orchestration Frameworks Comparison

| Framework | Best For | Language | UI | Complexity |
|-----------|----------|----------|-----|-----------|
| **Apache Airflow** | Complex workflows | Python | ✅ Excellent | Medium |
| **Prefect** | Modern pipelines | Python | ✅ Good | Low |
| **Kubeflow Pipelines** | Kubernetes ML | Python | ✅ Good | High |
| **AWS Step Functions** | AWS-native | JSON/Python | ✅ Basic | Medium |
| **Azure ML Pipelines** | Azure-native | Python | ✅ Good | Medium |
| **MLflow Projects** | Simple workflows | Any | ❌ No | Low |
| **Dagster** | Data pipelines | Python | ✅ Excellent | Medium |
| **Argo Workflows** | Kubernetes | YAML | ✅ Good | High |

## 📅 Scheduling Frameworks

### Apache Airflow (Most Popular)

**Architecture:**
```
┌─────────────────────────────────────────────┐
│           APACHE AIRFLOW                     │
├─────────────────────────────────────────────┤
│                                             │
│  Web UI ──► Scheduler ──► Executor         │
│              │                │              │
│              ├─► DAG Parser   │              │
│              │                ▼              │
│              └──────────► Workers           │
│                            (Tasks)           │
│                                             │
│  Metadata Database (PostgreSQL)             │
│   ├─ DAG runs                               │
│   ├─ Task instances                         │
│   └─ Logs                                   │
└─────────────────────────────────────────────┘
```

**Example DAG:**
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'ml-team',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'ml_training_pipeline',
    default_args=default_args,
    schedule_interval='0 2 * * 1',  # Monday 2 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:

    ingest_data = PythonOperator(
        task_id='ingest_data',
        python_callable=ingest_data_func,
    )

    validate_data = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data_func,
    )

    train_model = PythonOperator(
        task_id='train_model',
        python_callable=train_model_func,
    )

    deploy_model = PythonOperator(
        task_id='deploy_model',
        python_callable=deploy_model_func,
    )

    # Define dependencies
    ingest_data >> validate_data >> train_model >> deploy_model
```

**Scheduling Options:**
```python
# Cron expressions
schedule_interval='0 0 * * *'      # Daily at midnight
schedule_interval='0 2 * * 1'      # Weekly on Monday 2 AM
schedule_interval='0 */6 * * *'    # Every 6 hours
schedule_interval='@daily'         # Daily
schedule_interval='@weekly'        # Weekly
schedule_interval='@hourly'        # Hourly
schedule_interval=None             # Manual trigger only

# Timedelta
schedule_interval=timedelta(hours=12)  # Every 12 hours
schedule_interval=timedelta(days=7)    # Weekly
```

### Prefect (Modern Alternative)

**Architecture:**
```
┌─────────────────────────────────────────────┐
│              PREFECT                         │
├─────────────────────────────────────────────┤
│                                             │
│  UI ──► Orchestrator ──► Agents            │
│          │                   │              │
│          ├─► Flow Runs       ▼              │
│          │                Workers           │
│          └─► Logs                           │
│                                             │
│  Database (PostgreSQL/SQLite)               │
└─────────────────────────────────────────────┘
```

**Example Flow:**
```python
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from datetime import timedelta

@task(retries=2, retry_delay_seconds=60)
def ingest_data():
    # Data ingestion logic
    return data

@task
def validate_data(data):
    # Validation logic
    return validated_data

@task
def train_model(data):
    # Training logic
    return model

@task
def deploy_model(model):
    # Deployment logic
    return endpoint_url

@flow(name="ml-training-pipeline")
def ml_pipeline():
    data = ingest_data()
    validated = validate_data(data)
    model = train_model(validated)
    endpoint = deploy_model(model)
    return endpoint

# Schedule the flow
if __name__ == "__main__":
    ml_pipeline.serve(
        name="ml-training-deployment",
        cron="0 2 * * 1",  # Monday 2 AM
    )
```

### AWS Step Functions (Cloud-Native)

**Example State Machine:**
```json
{
  "StartAt": "IngestData",
  "States": {
    "IngestData": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:us-east-1:123456789012:function:IngestData",
      "Next": "ValidateData"
    },
    "ValidateData": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:us-east-1:123456789012:function:ValidateData",
      "Next": "TrainModel"
    },
    "TrainModel": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sagemaker:createTrainingJob.sync",
      "Parameters": {
        "TrainingJobName.$": "$.trainingJobName",
        "RoleArn": "arn:aws:iam::123456789012:role/SageMakerRole"
      },
      "Next": "EvaluateModel"
    },
    "EvaluateModel": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:us-east-1:123456789012:function:EvaluateModel",
      "Next": "DeploymentGate"
    },
    "DeploymentGate": {
      "Type": "Choice",
      "Choices": [
        {
          "Variable": "$.metrics.accuracy",
          "NumericGreaterThan": 0.85,
          "Next": "DeployModel"
        }
      ],
      "Default": "SendAlert"
    },
    "DeployModel": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sagemaker:createEndpoint",
      "End": true
    },
    "SendAlert": {
      "Type": "Task",
      "Resource": "arn:aws:sns:us-east-1:123456789012:ml-alerts",
      "End": true
    }
  }
}
```

## 🎯 Complete Pipeline Example

### End-to-End ML Pipeline with Airflow

```python
"""
Complete ML Pipeline DAG
Runs weekly to retrain model with latest data
"""

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import mlflow
from transformers import AutoModelForSequenceClassification

# Configuration
MLFLOW_TRACKING_URI = "http://mlflow:5000"
MODEL_NAME = "customer-sentiment-classifier"
MIN_ACCURACY = 0.85

def task_ingest_data(**context):
    """Ingest data from database"""
    import psycopg2
    import pandas as pd

    conn = psycopg2.connect("postgresql://user:pass@db:5432/production")
    query = "SELECT text, sentiment FROM customer_feedback WHERE created_at > NOW() - INTERVAL '7 days'"
    df = pd.read_sql(query, conn)

    # Save to shared location
    df.to_parquet('/tmp/raw_data.parquet')

    return {'row_count': len(df)}

def task_validate_data(**context):
    """Validate data quality"""
    import pandas as pd
    from great_expectations.dataset import PandasDataset

    df = pd.read_parquet('/tmp/raw_data.parquet')
    dataset = PandasDataset(df)

    # Define expectations
    assert dataset.expect_column_to_exist('text').success
    assert dataset.expect_column_values_to_not_be_null('text').success
    assert dataset.expect_column_values_to_be_in_set('sentiment', ['positive', 'negative', 'neutral']).success

    print(f"✓ Validated {len(df)} rows")

def task_preprocess_data(**context):
    """Preprocess and split data"""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_parquet('/tmp/raw_data.parquet')

    # Encode labels
    label_map = {'positive': 0, 'neutral': 1, 'negative': 2}
    df['label'] = df['sentiment'].map(label_map)

    # Split
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'])

    train_df.to_parquet('/tmp/train_data.parquet')
    test_df.to_parquet('/tmp/test_data.parquet')

    return {'train_size': len(train_df), 'test_size': len(test_df)}

def task_train_model(**context):
    """Train model with transfer learning"""
    import pandas as pd
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("customer-sentiment")

    with mlflow.start_run():
        # Load pre-trained BERT
        model = AutoModelForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=3
        )
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

        # Load data
        train_df = pd.read_parquet('/tmp/train_data.parquet')

        # Tokenize
        train_encodings = tokenizer(train_df['text'].tolist(), truncation=True, padding=True)

        # Training arguments
        training_args = TrainingArguments(
            output_dir='/tmp/model',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            learning_rate=2e-5,
            logging_steps=100,
        )

        # Train
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_encodings,
        )

        trainer.train()

        # Log to MLflow
        mlflow.log_params({'model': 'distilbert-base-uncased', 'epochs': 3})

        # Save model
        model.save_pretrained('/tmp/trained_model')

        return mlflow.active_run().info.run_id

def task_evaluate_model(**context):
    """Evaluate model"""
    import pandas as pd
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from sklearn.metrics import accuracy_score, f1_score

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained('/tmp/trained_model')
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    # Load test data
    test_df = pd.read_parquet('/tmp/test_data.parquet')

    # Predict
    test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True, return_tensors='pt')
    outputs = model(**test_encodings)
    predictions = outputs.logits.argmax(dim=1).numpy()

    # Metrics
    accuracy = accuracy_score(test_df['label'], predictions)
    f1 = f1_score(test_df['label'], predictions, average='weighted')

    # Log to MLflow
    mlflow.log_metrics({'accuracy': accuracy, 'f1_score': f1})

    # Save metrics
    metrics = {'accuracy': accuracy, 'f1_score': f1}

    import json
    with open('/tmp/metrics.json', 'w') as f:
        json.dump(metrics, f)

    return metrics

def task_quality_gate(**context):
    """Decide if model should be deployed"""
    import json

    with open('/tmp/metrics.json', 'r') as f:
        metrics = json.load(f)

    if metrics['accuracy'] >= MIN_ACCURACY:
        return 'deploy_model'
    else:
        return 'send_failure_alert'

def task_deploy_model(**context):
    """Deploy model to production"""
    # Register in MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    run_id = context['ti'].xcom_pull(task_ids='train_model')

    # Register model
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri, MODEL_NAME)

    # Transition to Production
    client = mlflow.tracking.MlflowClient()
    latest_version = client.get_latest_versions(MODEL_NAME, stages=["None"])[0].version
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=latest_version,
        stage="Production"
    )

    print(f"✓ Model {MODEL_NAME} v{latest_version} deployed to Production")

def task_send_success_alert(**context):
    """Send success notification"""
    print("✉️  Model deployed successfully!")
    # Send Slack/email notification

def task_send_failure_alert(**context):
    """Send failure notification"""
    print("✉️  Model did not meet quality threshold!")
    # Send Slack/email alert

# Define DAG
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email': ['ml-team@company.com'],
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'ml_retraining_pipeline',
    default_args=default_args,
    description='Weekly ML model retraining with transfer learning',
    schedule_interval='0 2 * * 1',  # Monday 2 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'production', 'transfer-learning'],
) as dag:

    # Define tasks
    ingest = PythonOperator(task_id='ingest_data', python_callable=task_ingest_data)
    validate = PythonOperator(task_id='validate_data', python_callable=task_validate_data)
    preprocess = PythonOperator(task_id='preprocess_data', python_callable=task_preprocess_data)
    train = PythonOperator(task_id='train_model', python_callable=task_train_model)
    evaluate = PythonOperator(task_id='evaluate_model', python_callable=task_evaluate_model)
    gate = BranchPythonOperator(task_id='quality_gate', python_callable=task_quality_gate)
    deploy = PythonOperator(task_id='deploy_model', python_callable=task_deploy_model)
    success_alert = PythonOperator(task_id='send_success_alert', python_callable=task_send_success_alert)
    failure_alert = PythonOperator(task_id='send_failure_alert', python_callable=task_send_failure_alert)

    # Define dependencies
    ingest >> validate >> preprocess >> train >> evaluate >> gate
    gate >> deploy >> success_alert
    gate >> failure_alert
```

## 🏭 Production Validation & Argo Rollouts

### Why Production Validation?

Before promoting a model to serve live traffic, it must pass a battery of automated checks against the **actual EKS cluster** it will run on. These checks ensure the model meets latency SLAs, can handle the required TPS, and doesn't regress on accuracy.

```text
┌─────────────────────────────────────────────────────────────────┐
│             PRODUCTION VALIDATION WORKFLOW                       │
└─────────────────────────────────────────────────────────────────┘

  ┌──────────┐     ┌──────────────┐     ┌──────────────────┐
  │  Model   │────►│  Build       │────►│  Deploy to       │
  │  trained │     │  container   │     │  staging on EKS  │
  └──────────┘     └──────────────┘     └────────┬─────────┘
                                                 │
                              ┌───────────────────┘
                              ▼
                   ┌──────────────────────┐
                   │  VALIDATION SUITE    │
                   │                      │
                   │  1. Health check     │
                   │  2. Accuracy / F1    │
                   │  3. p50/p95/p99      │
                   │  4. Sustained TPS    │
                   │  5. Error rate       │
                   │  6. Model size / mem │
                   │  7. EKS capacity     │
                   └──────────┬───────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
               ALL PASS            ANY FAIL
                    │                   │
                    ▼                   ▼
          ┌─────────────────┐  ┌────────────────┐
          │ Argo Rollouts   │  │ Rollout paused │
          │ promotion       │  │ team alerted   │
          │ (canary / B-G)  │  │ fix & re-run   │
          └─────────────────┘  └────────────────┘
```

### Validation Suite Checks

| Check | What it measures | Typical threshold |
|-------|-----------------|-------------------|
| **Accuracy** | Classification accuracy on held-out set | >= 0.85 |
| **F1 Score** | Weighted F1 across classes | >= 0.82 |
| **p50 Latency** | Median response time | <= 50 ms |
| **p95 Latency** | 95th percentile response time | <= 150 ms |
| **p99 Latency** | 99th percentile response time | <= 300 ms |
| **Sustained TPS** | Transactions/sec under load | >= 500 req/s |
| **Error Rate** | Failed requests / total requests | <= 1% |
| **Model Size** | Serialised model on disk | <= 500 MB |
| **Memory Usage** | RSS during inference | <= 512 MB |
| **EKS Capacity** | Cluster can host required replicas | CPU + RAM fit |

```python
# Run the validation suite directly:
#   python code_folder/19_model_production_validation.py
#
# Or import it programmatically (filename starts with a digit):
import importlib.util, sys
spec = importlib.util.spec_from_file_location(
    "model_production_validation",
    "code_folder/19_model_production_validation.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

cluster = mod.EKSClusterSpec(node_count=3, node_instance_type="m5.2xlarge")
thresholds = mod.ValidationThresholds(min_accuracy=0.85, min_tps=500)
endpoint = mod.SimulatedModelEndpoint(accuracy=0.92)

validator = mod.ModelProductionValidator(endpoint, thresholds, cluster)
report = validator.run_full_suite("my-model", "2.1.0")

if report.all_passed:
    print("Promote via Argo Rollouts")
```

### Load Testing for TPS Assurance

Before promotion, a multi-phase load test ramps concurrency against the staging endpoint:

```text
Phase 1 (warm-up):   5 workers  × 10s
Phase 2 (ramp):      10 workers × 15s
Phase 3 (target):    25 workers × 20s   ← evaluated against thresholds
Phase 4 (peak):      50 workers × 20s   ← stress ceiling
Phase 5 (cool-down): 10 workers × 10s
```

The load test can run as a Kubernetes Job on the same EKS cluster:

```bash
kubectl apply -f load-test-job.yaml
kubectl wait --for=condition=complete job/ml-model-load-test -n ml-serving
```

See `code_folder/19_model_production_load_test.py` for the full harness.

### Argo Rollouts - Canary Deployment

Canary gradually shifts traffic from the stable model to the new version. At each step, an `AnalysisTemplate` queries Prometheus to verify the promotion criteria are met.

```text
Traffic flow with canary:

  100% stable                 10% canary              30% canary
  ┌──────────┐     ┌──────────┬──────────┐     ┌──────┬──────────┐
  │  v1.0.0  │ ──► │  v1.0.0  │  v2.1.0  │ ──► │v1.0.0│  v2.1.0  │
  │  (prod)  │     │  90%     │  10%     │     │ 70%  │  30%     │
  └──────────┘     └──────────┴──────────┘     └──────┴──────────┘
                        │ analysis pass              │ analysis pass
                        ▼                            ▼
                   60% canary                   100% promoted
              ┌──────┬──────────┐     ┌──────────────────────┐
              │v1.0.0│  v2.1.0  │ ──► │       v2.1.0         │
              │ 40%  │  60%     │     │   (new stable)       │
              └──────┴──────────┘     └──────────────────────┘
```

```yaml
# Argo Rollout canary steps (see argo-rollouts/canary-rollout.yaml)
steps:
  - setWeight: 10
  - analysis:
      templates:
        - templateName: ml-model-analysis   # checks TPS, latency, accuracy
  - pause: { duration: 2m }
  - setWeight: 30
  - analysis:
      templates:
        - templateName: ml-model-analysis
  - pause: { duration: 3m }
  - setWeight: 60
  - analysis:
      templates:
        - templateName: ml-model-analysis
  - pause: { duration: 5m }
  - setWeight: 100
```

**Promotion criteria (AnalysisTemplate):**
- Success rate >= 99% (Prometheus: `model_inference_requests_total`)
- p95 latency <= 150ms (Prometheus: `model_inference_latency_seconds_bucket`)
- Throughput >= 200 TPS (Prometheus: `model_inference_requests_total` rate)
- Model accuracy >= 85% (Prometheus: `model_inference_accuracy`)

If **any** metric breaches its threshold, the rollout automatically aborts and traffic reverts to the stable version.

### Argo Rollouts - Blue-Green Deployment

Blue-green keeps the current (active) version running while the new (preview) version is validated in parallel. Traffic switches all at once after analysis passes.

```text
Blue-Green flow:

  ┌──────────────┐      ┌──────────────┐     ┌──────────────┐
  │   Active     │      │   Active     │     │   Active     │
  │   v1.0.0     │      │   v1.0.0     │     │   v2.1.0     │
  │  (live)      │      │  (live)      │     │  (promoted)  │
  └──────────────┘      └──────────────┘     └──────────────┘
                        ┌──────────────┐
                        │   Preview    │      old pods scaled
                        │   v2.1.0    │      down after 30s
                        │  (testing)   │
                        └──────┬───────┘
                               │
                      pre-promotion analysis
                        (TPS, latency, accuracy)
                               │
                          pass ──► switch
                          fail ──► abort + rollback
```

```bash
# Commands
kubectl argo rollouts get rollout ml-model-bluegreen --watch
kubectl argo rollouts promote ml-model-bluegreen       # manual promote
kubectl argo rollouts abort ml-model-bluegreen          # manual abort
```

### Choosing Canary vs Blue-Green

| Aspect | Canary | Blue-Green |
|--------|--------|------------|
| **Traffic shift** | Gradual (10% → 30% → 60% → 100%) | All-at-once |
| **Risk** | Lower (small blast radius) | Higher (full switch) |
| **Rollback speed** | Instant (shift weight back) | Instant (switch service) |
| **Resource cost** | Lower (shared pods) | Higher (2x pods during switch) |
| **Best for** | High-traffic endpoints, risk-averse | Fast promotion, simpler testing |
| **Analysis points** | Multiple (each step) | Two (pre + post promotion) |

### CI/CD Integration

The complete automated pipeline:

```text
┌─────────────┐   ┌──────────────┐   ┌─────────────────┐   ┌──────────────┐
│ Model train │──►│ Register in  │──►│ Build container  │──►│ Push to ECR  │
│ (MLflow)    │   │ MLflow       │   │ (Docker/Kaniko)  │   │              │
└─────────────┘   └──────────────┘   └─────────────────┘   └──────┬───────┘
                                                                  │
                  ┌──────────────────────────────────────────────┘
                  ▼
         ┌─────────────────┐   ┌──────────────────┐   ┌────────────────┐
         │ Update Rollout  │──►│ Validation suite  │──►│ Load test Job  │
         │ image tag       │   │ (accuracy, F1)    │   │ (TPS, latency) │
         └─────────────────┘   └──────────────────┘   └───────┬────────┘
                                                              │
                                                    ┌─────────┴─────────┐
                                                    │                   │
                                               ALL PASS            ANY FAIL
                                                    │                   │
                                                    ▼                   ▼
                                         ┌──────────────────┐  ┌──────────────┐
                                         │ argo rollouts    │  │ argo rollouts│
                                         │ promote          │  │ abort        │
                                         └──────────────────┘  └──────────────┘
```

### Related Files

- [`code_folder/19_model_production_validation.py`](code_folder/19_model_production_validation.py) - Full validation suite
- [`code_folder/19_model_production_load_test.py`](code_folder/19_model_production_load_test.py) - TPS / latency load harness
- [`code_folder/basics/kubernetes/argo-rollouts/canary-rollout.yaml`](code_folder/basics/kubernetes/argo-rollouts/canary-rollout.yaml) - Canary with AnalysisTemplate
- [`code_folder/basics/kubernetes/argo-rollouts/blue-green-rollout.yaml`](code_folder/basics/kubernetes/argo-rollouts/blue-green-rollout.yaml) - Blue-green with pre/post analysis

## 📚 Best Practices

### Model Development
- ✅ Start with pre-trained models when possible
- ✅ Use transfer learning for faster development
- ✅ Track all experiments with MLflow
- ✅ Version datasets and models
- ✅ Implement comprehensive testing

### Pipeline Design
- ✅ Make tasks idempotent (can re-run safely)
- ✅ Use quality gates before deployment
- ✅ Implement retry logic with exponential backoff
- ✅ Add monitoring and alerting
- ✅ Document task dependencies

### Deployment
- ✅ Use canary deployments for safety
- ✅ Monitor model performance continuously
- ✅ Implement automated rollback on degradation
- ✅ Version all deployed models
- ✅ Maintain deployment documentation

### Scheduling
- ✅ Schedule retraining based on data freshness needs
- ✅ Use off-peak hours for heavy workloads
- ✅ Implement SLA monitoring
- ✅ Set up failure notifications
- ✅ Plan for backfills and catch-ups

## 🔗 Related Documentation

- [MLOPS_INFRASTRUCTURE.md](MLOPS_INFRASTRUCTURE.md) - Infrastructure setup
- [code_folder/19_airflow_ml_pipeline.py](code_folder/19_airflow_ml_pipeline.py) - Complete Airflow example
- [code_folder/21_transfer_learning_pretrained_models.py](code_folder/21_transfer_learning_pretrained_models.py) - Transfer learning guide
- [code_folder/19_mlflow_model_registry.py](code_folder/19_mlflow_model_registry.py) - Model versioning
- [code_folder/19_model_production_validation.py](code_folder/19_model_production_validation.py) - Production validation suite
- [code_folder/19_model_production_load_test.py](code_folder/19_model_production_load_test.py) - TPS load testing harness
- [code_folder/basics/kubernetes/argo-rollouts/](code_folder/basics/kubernetes/argo-rollouts/) - Argo Rollouts manifests

## 🎓 Learning Path

1. **Week 1-2**: Understand transfer learning
   - Run `21_transfer_learning_pretrained_models.py`
   - Practice fine-tuning BERT and ResNet

2. **Week 3-4**: Learn pipeline orchestration
   - Run `19_airflow_ml_pipeline.py`
   - Create simple Airflow DAGs

3. **Week 5-6**: Implement complete lifecycle
   - Combine transfer learning + Airflow
   - Deploy to local environment

4. **Week 7-8**: Production validation & deployment
   - Run `19_model_production_validation.py` for pre-deployment gates
   - Run `19_model_production_load_test.py` for TPS assurance
   - Deploy Argo Rollouts canary/blue-green on EKS
   - Monitor and iterate

## 💡 Quick Reference

### Load Pre-trained Model
```python
from transformers import AutoModel
model = AutoModel.from_pretrained('bert-base-uncased')
```

### Schedule Airflow DAG
```python
schedule_interval='0 2 * * 1'  # Monday 2 AM
```

### Track with MLflow
```python
import mlflow
with mlflow.start_run():
    mlflow.log_param("lr", 0.01)
    mlflow.log_metric("accuracy", 0.95)
```

### Deploy with Quality Gate
```python
if accuracy >= MIN_THRESHOLD:
    deploy_model()
else:
    send_alert()
```
