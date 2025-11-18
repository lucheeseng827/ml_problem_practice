# Complete ML Model Lifecycle with Pre-trained Models and Pipeline Orchestration

This guide explains the complete lifecycle of ML models from development to production, with focus on using pre-trained models and orchestration frameworks.

## 📋 Table of Contents

- [Model Lifecycle Overview](#model-lifecycle-overview)
- [Using Pre-trained Models](#using-pre-trained-models)
- [Pipeline Orchestration](#pipeline-orchestration)
- [Scheduling Frameworks](#scheduling-frameworks)
- [Complete Examples](#complete-examples)

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

4. **Week 7-8**: Production deployment
   - Deploy to AWS with Terraform
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
