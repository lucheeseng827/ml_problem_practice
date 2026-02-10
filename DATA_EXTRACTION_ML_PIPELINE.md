# Data Extraction & ML Pipeline Practice

A hands-on guide covering end-to-end data extraction, data engineering, ML model training, inference, and overlay on test datasets — using public data sources. Includes instructions for running each stage in Amazon SageMaker.

> **Scope**: This guide covers the **code and pipeline logic** only. Cloud infrastructure (SageMaker endpoints, Redshift clusters, IAM roles, VPCs) is managed separately via Terraform/CloudFormation and is not covered here.

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Public Data Sources](#2-public-data-sources)
3. [Stage 1 — Data Extraction](#3-stage-1--data-extraction)
4. [Stage 2 — Data Engineering](#4-stage-2--data-engineering)
5. [Stage 3 — ML Model Training](#5-stage-3--ml-model-training)
6. [Stage 4 — Inference & Overlay on Test Dataset](#6-stage-4--inference--overlay-on-test-dataset)
7. [Running in SageMaker](#7-running-in-sagemaker)
8. [Exercise List](#8-exercise-list)
9. [References](#9-references)

---

## 1. Pipeline Overview

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────────────┐
│   Extract    │───▶│   Engineer   │───▶│   Train ML    │───▶│  Inference &     │
│   Data       │    │   Data       │    │   Model       │    │  Overlay on Test │
└─────────────┘    └──────────────┘    └───────────────┘    └──────────────────┘
  - Public APIs      - Clean nulls      - Train/val split    - Load held-out test
  - CSV/Parquet      - Type casting      - Feature select     - Run predictions
  - SQL databases    - Feature eng.      - Fit model          - Overlay predictions
  - Web scraping     - Normalization     - Evaluate metrics     on test dataframe
                     - Encoding          - Save artifacts      - Export results
```

**Scripts** (in `code_folder/`):

| File | Purpose |
|------|---------|
| `22_data_extraction_pipeline.py` | Extract data from public APIs, CSV URLs, and SQLite |
| `22_data_engineering_pipeline.py` | Clean, transform, and feature-engineer extracted data |
| `22_ml_model_training.py` | Train a simple ML model with evaluation |
| `22_ml_inference_overlay.py` | Run inference on test set and overlay predictions |
| `22_sagemaker_pipeline_runner.py` | Orchestrate all stages as a SageMaker Pipeline |

---

## 2. Public Data Sources

| Dataset | Source | Format | Use Case |
|---------|--------|--------|----------|
| UCI Adult Income | [UCI ML Repository](https://archive.ics.uci.edu/dataset/2/adult) | CSV | Binary classification |
| California Housing | `sklearn.datasets` | In-memory | Regression |
| Titanic | [Stanford CS](https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv) | CSV | Binary classification |
| NYC Taxi Trip Data | [NYC TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) | Parquet | Regression / Time series |
| OpenWeather API | [openweathermap.org](https://openweathermap.org/api) | JSON API | Feature enrichment |
| World Bank Indicators | [World Bank API](https://api.worldbank.org/v2/) | JSON API | Macro-economic features |

---

## 3. Stage 1 — Data Extraction

**Script**: `code_folder/22_data_extraction_pipeline.py`

### Techniques Covered

#### 3.1 Extract from CSV URL
```python
import pandas as pd

url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
df = pd.read_csv(url)
```

#### 3.2 Extract from REST API
```python
import requests

response = requests.get(
    "https://api.worldbank.org/v2/country/US/indicator/NY.GDP.MKTP.CD",
    params={"format": "json", "date": "2015:2023", "per_page": 100},
)
records = response.json()[1]
df = pd.json_normalize(records)
```

#### 3.3 Extract from Parquet (e.g., NYC Taxi)
```python
url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"
df = pd.read_parquet(url)
```

#### 3.4 Extract from SQLite (local database simulation)
```python
import sqlite3

conn = sqlite3.connect("extracted_data.db")
df.to_sql("raw_titanic", conn, if_exists="replace", index=False)
df_back = pd.read_sql("SELECT * FROM raw_titanic", conn)
```

#### 3.5 Extract from sklearn Built-in Datasets
```python
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
df = housing.frame
```

### Validation After Extraction
```python
assert df.shape[0] > 0, "No rows extracted"
assert df.shape[1] > 0, "No columns extracted"
print(f"Extracted {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")
print(f"Null counts:\n{df.isnull().sum()}")
```

---

## 4. Stage 2 — Data Engineering

**Script**: `code_folder/22_data_engineering_pipeline.py`

### Techniques Covered

#### 4.1 Handle Missing Values
```python
# Numeric: fill with median
numeric_cols = df.select_dtypes(include="number").columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Categorical: fill with mode
cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
```

#### 4.2 Type Casting and Cleanup
```python
df["Age"] = df["Age"].astype(float)
df["Pclass"] = df["Pclass"].astype("category")

# Strip whitespace from string columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()
```

#### 4.3 Feature Engineering
```python
# Binning continuous variables
df["Age_Group"] = pd.cut(df["Age"], bins=[0, 12, 18, 35, 60, 100],
                         labels=["Child", "Teen", "Adult", "Middle", "Senior"])

# Interaction features
df["Fare_Per_Person"] = df["Fare"] / (df["Siblings/Spouses Aboard"] + 1)

# Log transform for skewed features
import numpy as np
df["Log_Fare"] = np.log1p(df["Fare"])
```

#### 4.4 Encoding Categorical Variables
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Label encoding for binary columns
le = LabelEncoder()
df["Sex_Encoded"] = le.fit_transform(df["Sex"])

# One-hot encoding for multi-class columns
df = pd.get_dummies(df, columns=["Age_Group"], prefix="Age", drop_first=True)
```

#### 4.5 Normalization / Scaling
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()
df[["Fare_Scaled"]] = scaler.fit_transform(df[["Fare"]])
```

#### 4.6 Train / Test Split with Stratification
```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

---

## 5. Stage 3 — ML Model Training

**Script**: `code_folder/22_ml_model_training.py`

### Techniques Covered

#### 5.1 Baseline Model (Logistic Regression)
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_train)
print(f"Train Accuracy: {accuracy_score(y_train, y_pred):.4f}")
```

#### 5.2 Gradient Boosted Model (XGBoost)
```python
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss",
)
xgb_model.fit(X_train, y_train)
```

#### 5.3 Cross-Validation
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

#### 5.4 Save Model Artifacts
```python
import joblib

joblib.dump(model, "model_artifacts/logistic_model.joblib")
joblib.dump(xgb_model, "model_artifacts/xgb_model.joblib")
joblib.dump(scaler, "model_artifacts/scaler.joblib")
joblib.dump(le, "model_artifacts/label_encoder.joblib")
```

---

## 6. Stage 4 — Inference & Overlay on Test Dataset

**Script**: `code_folder/22_ml_inference_overlay.py`

### Techniques Covered

#### 6.1 Load Model and Run Inference
```python
import joblib
import pandas as pd

model = joblib.load("model_artifacts/xgb_model.joblib")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
```

#### 6.2 Overlay Predictions on Test DataFrame
```python
test_results = X_test.copy()
test_results["Actual"] = y_test.values
test_results["Predicted"] = y_pred
test_results["Predicted_Probability"] = y_proba
test_results["Correct"] = test_results["Actual"] == test_results["Predicted"]
```

#### 6.3 Evaluate on Test Set
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
)

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_proba),
}
```

#### 6.4 Export Overlayed Results
```python
test_results.to_csv("output/test_predictions_overlay.csv", index=False)
test_results.to_parquet("output/test_predictions_overlay.parquet", index=False)

pd.DataFrame([metrics]).to_csv("output/test_metrics.csv", index=False)
```

#### 6.5 Error Analysis
```python
errors = test_results[~test_results["Correct"]]
print(f"Total errors: {len(errors)} / {len(test_results)}")
print(f"Error rate: {len(errors) / len(test_results):.2%}")
print(f"\nError distribution by class:\n{errors['Actual'].value_counts()}")
```

---

## 7. Running in SageMaker

> Infrastructure (SageMaker domain, IAM roles, S3 buckets) is already provisioned. The instructions below cover **how to run the pipeline code** inside SageMaker.

### 7.1 SageMaker Studio Notebook (Interactive)

Use SageMaker Studio notebooks for development and experimentation.

```python
# In a SageMaker Studio notebook cell:
import sagemaker
from sagemaker.session import Session

session = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = session.default_bucket()
prefix = "ml-pipeline-practice"
```

**Upload data to S3**:
```python
import boto3

s3 = boto3.client("s3")
s3.upload_file("data/processed_train.csv", bucket, f"{prefix}/data/train.csv")
s3.upload_file("data/processed_test.csv", bucket, f"{prefix}/data/test.csv")
```

**Run scripts directly** in notebook cells or via `%run`:
```python
%run code_folder/22_data_extraction_pipeline.py
%run code_folder/22_data_engineering_pipeline.py
%run code_folder/22_ml_model_training.py
%run code_folder/22_ml_inference_overlay.py
```

### 7.2 SageMaker Processing Jobs (Batch Execution)

Use Processing Jobs to run extraction and engineering steps at scale.

```python
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearnProcessor

sklearn_processor = SKLearnProcessor(
    framework_version="1.2-1",
    role=role,
    instance_type="ml.m5.xlarge",
    instance_count=1,
    base_job_name="data-extraction",
)

# Run data extraction
sklearn_processor.run(
    code="code_folder/22_data_extraction_pipeline.py",
    outputs=[
        ProcessingOutput(
            output_name="extracted",
            source="/opt/ml/processing/output",
            destination=f"s3://{bucket}/{prefix}/extracted/",
        )
    ],
)

# Run data engineering
sklearn_processor.run(
    code="code_folder/22_data_engineering_pipeline.py",
    inputs=[
        ProcessingInput(
            source=f"s3://{bucket}/{prefix}/extracted/",
            destination="/opt/ml/processing/input",
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="engineered",
            source="/opt/ml/processing/output",
            destination=f"s3://{bucket}/{prefix}/engineered/",
        )
    ],
)
```

### 7.3 SageMaker Training Jobs

Use built-in or custom training for the model training step.

```python
from sagemaker.sklearn import SKLearn

estimator = SKLearn(
    entry_point="22_ml_model_training.py",
    source_dir="code_folder",
    role=role,
    instance_type="ml.m5.xlarge",
    instance_count=1,
    framework_version="1.2-1",
    hyperparameters={
        "n-estimators": 200,
        "max-depth": 5,
        "learning-rate": 0.1,
    },
    output_path=f"s3://{bucket}/{prefix}/model/",
)

estimator.fit({
    "train": f"s3://{bucket}/{prefix}/engineered/train/",
    "test": f"s3://{bucket}/{prefix}/engineered/test/",
})
```

### 7.4 SageMaker Batch Transform (Inference at Scale)

Run inference on the full test dataset without deploying an endpoint.

```python
transformer = estimator.transformer(
    instance_count=1,
    instance_type="ml.m5.xlarge",
    output_path=f"s3://{bucket}/{prefix}/predictions/",
)

transformer.transform(
    data=f"s3://{bucket}/{prefix}/engineered/test/",
    content_type="text/csv",
    split_type="Line",
)
transformer.wait()
```

### 7.5 SageMaker Pipelines (Full Orchestration)

Chain all stages into a single reproducible pipeline.

```python
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, TransformStep

# See code_folder/22_sagemaker_pipeline_runner.py for the full implementation

pipeline = Pipeline(
    name="data-extraction-ml-pipeline",
    steps=[extract_step, engineer_step, train_step, inference_step],
    sagemaker_session=session,
)

pipeline.upsert(role_arn=role)
execution = pipeline.start()
execution.wait()
```

### 7.6 SageMaker Experiments (Tracking)

Track each run for reproducibility.

```python
from sagemaker.experiments import Run

with Run(experiment_name="extraction-ml-pipeline", run_name="run-001") as run:
    run.log_parameter("model_type", "xgboost")
    run.log_parameter("n_estimators", 200)
    # ... train model ...
    run.log_metric("accuracy", metrics["accuracy"])
    run.log_metric("f1_score", metrics["f1"])
    run.log_metric("roc_auc", metrics["roc_auc"])
```

### SageMaker Execution Summary

| Pipeline Stage | SageMaker Feature | Instance Recommendation |
|----------------|-------------------|------------------------|
| Data Extraction | Processing Job (SKLearnProcessor) | ml.m5.xlarge |
| Data Engineering | Processing Job (SKLearnProcessor) | ml.m5.xlarge |
| Model Training | Training Job (SKLearn Estimator) | ml.m5.xlarge (CPU) or ml.g4dn.xlarge (GPU) |
| Inference / Overlay | Batch Transform | ml.m5.xlarge |
| Full Pipeline | SageMaker Pipelines | Per-step config |
| Experiment Tracking | SageMaker Experiments | N/A (metadata only) |

---

## 8. Exercise List

### Beginner

| # | Exercise | Key Concepts |
|---|----------|-------------|
| 1 | Extract Titanic CSV from URL, print shape and dtypes | `pd.read_csv()`, URL fetch |
| 2 | Extract California Housing from sklearn, save to CSV | `fetch_california_housing()`, `to_csv()` |
| 3 | Load CSV into SQLite, query back with SQL filter | `sqlite3`, `to_sql()`, `read_sql()` |
| 4 | Fill missing numeric values with median | `fillna()`, `median()` |
| 5 | One-hot encode a categorical column | `pd.get_dummies()` |
| 6 | Standard-scale numeric features | `StandardScaler`, `fit_transform()` |
| 7 | Train a Logistic Regression, print accuracy | `LogisticRegression`, `accuracy_score()` |
| 8 | Generate predictions on test set, overlay on dataframe | `predict()`, `df.assign()` |

### Intermediate

| # | Exercise | Key Concepts |
|---|----------|-------------|
| 9 | Extract from a REST API (World Bank), normalize JSON | `requests.get()`, `json_normalize()` |
| 10 | Read Parquet file from URL, filter and sample | `pd.read_parquet()`, `query()`, `sample()` |
| 11 | Build a feature engineering pipeline with `ColumnTransformer` | `Pipeline`, `ColumnTransformer` |
| 12 | Train XGBoost with cross-validation, report mean CV score | `XGBClassifier`, `cross_val_score()` |
| 13 | Compare Logistic Regression vs XGBoost on same test set | Side-by-side metrics, overlay both |
| 14 | Save and reload model artifacts with joblib | `joblib.dump()`, `joblib.load()` |
| 15 | Compute confusion matrix and classification report | `confusion_matrix()`, `classification_report()` |
| 16 | Run data extraction as a SageMaker Processing Job | `SKLearnProcessor`, `ProcessingOutput` |

### Advanced

| # | Exercise | Key Concepts |
|---|----------|-------------|
| 17 | Build end-to-end pipeline: extract → engineer → train → infer | Multi-script orchestration |
| 18 | Extract from multiple sources, merge into single dataset | `pd.merge()`, API + CSV join |
| 19 | Implement data validation checks between stages | Schema checks, row count assertions |
| 20 | Overlay predictions with confidence intervals | `predict_proba()`, threshold analysis |
| 21 | Error analysis: segment misclassifications by feature group | Groupby on error dataframe |
| 22 | Orchestrate full pipeline with SageMaker Pipelines | `Pipeline`, `ProcessingStep`, `TrainingStep` |
| 23 | Track experiments with SageMaker Experiments | `Run`, `log_metric()`, `log_parameter()` |
| 24 | Run batch inference with SageMaker Batch Transform | `Transformer`, `transform()` |

---

## 9. References

- [SageMaker Python SDK — Processing](https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_processing.html)
- [SageMaker Python SDK — Training](https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/sagemaker.sklearn.html)
- [SageMaker Python SDK — Pipelines](https://sagemaker.readthedocs.io/en/stable/workflows/pipelines/index.html)
- [SageMaker Python SDK — Experiments](https://sagemaker.readthedocs.io/en/stable/experiments/sagemaker.experiments.html)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [UCI ML Repository](https://archive.ics.uci.edu/)
