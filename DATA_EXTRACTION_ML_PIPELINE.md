# Data Extraction & ML Pipeline Practice

A hands-on guide covering end-to-end data extraction, data engineering, ML model training, inference, and overlay on test datasets — using public data sources. Includes instructions for running each stage in Amazon SageMaker.

> **Scope**: This guide covers the **code and pipeline logic** only. Cloud infrastructure (SageMaker endpoints, Redshift clusters, IAM roles, VPCs) is managed separately via Terraform/CloudFormation and is not covered here.

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Public Data Sources](#2-public-data-sources)
3. [Stage 1 — Data Extraction](#3-stage-1--data-extraction)
4. [Stage 1B — Glue ETL, S3 Data Lake & Athena](#4-stage-1b--glue-etl-s3-data-lake--athena)
5. [Stage 2 — Data Engineering](#5-stage-2--data-engineering)
6. [Stage 3 — ML Model Training](#6-stage-3--ml-model-training)
7. [Stage 4 — Inference & Overlay on Test Dataset](#7-stage-4--inference--overlay-on-test-dataset)
8. [Running in SageMaker](#8-running-in-sagemaker)
9. [Exercise List](#9-exercise-list)
10. [References](#10-references)

---

## 1. Pipeline Overview

```
┌─────────────┐    ┌───────────────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────────────┐
│   Extract    │───▶│  Glue / S3 /      │───▶│   Engineer   │───▶│   Train ML    │───▶│  Inference &     │
│   Data       │    │  Athena           │    │   Data       │    │   Model       │    │  Overlay on Test │
└─────────────┘    └───────────────────┘    └──────────────┘    └───────────────┘    └──────────────────┘
  - Public APIs      - S3 data lake        - Clean nulls      - Train/val split    - Load held-out test
  - CSV/Parquet      - Glue Crawlers       - Type casting      - Feature select     - Run predictions
  - SQL databases    - Glue ETL jobs       - Feature eng.      - Fit model          - Overlay predictions
  - Web scraping     - Athena SQL queries  - Normalization     - Evaluate metrics     on test dataframe
                     - Downstream export   - Encoding          - Save artifacts      - Export results
```

**Scripts** (in `code_folder/`):

| File | Purpose |
|------|---------|
| `22_data_extraction_pipeline.py` | Extract data from public APIs, CSV URLs, and SQLite |
| `22_glue_s3_athena_pipeline.py` | S3 import, Glue ETL cleaning, Athena querying, downstream export |
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

## 4. Stage 1B — Glue ETL, S3 Data Lake & Athena

**Script**: `code_folder/22_glue_s3_athena_pipeline.py`

This stage lands extracted data into S3, catalogs it with Glue Crawlers, transforms it with Glue ETL jobs, queries it via Athena, and exports results downstream to other data stores.

> **Infra note**: Glue databases, IAM roles, and S3 buckets are provisioned via Terraform. This section covers the **pipeline code** only.

### Techniques Covered

#### 4.1 Upload Raw Data to S3 (Data Lake Ingestion)
```python
import boto3

s3 = boto3.client("s3")
bucket = "my-data-lake-bucket"
prefix = "raw/titanic"

# Upload CSV
s3.upload_file("data/extracted/titanic_raw.csv", bucket, f"{prefix}/titanic_raw.csv")

# Upload Parquet (columnar, better for Athena)
s3.upload_file("data/extracted/taxi_raw.parquet", bucket, "raw/taxi/taxi_raw.parquet")

# Upload partitioned data (year/month partitioning for Athena performance)
for year in [2022, 2023]:
    for month in range(1, 13):
        key = f"raw/events/year={year}/month={month:02d}/data.parquet"
        s3.upload_file(f"data/events_{year}_{month}.parquet", bucket, key)
```

#### 4.2 Glue Crawler — Auto-Discover Schema
```python
glue = boto3.client("glue")

# Create a crawler that infers schema from S3 data
glue.create_crawler(
    Name="titanic-raw-crawler",
    Role="arn:aws:iam::123456789012:role/GlueServiceRole",
    DatabaseName="ml_pipeline_db",
    Targets={
        "S3Targets": [
            {"Path": f"s3://{bucket}/raw/titanic/"},
        ]
    },
    TablePrefix="raw_",
    SchemaChangePolicy={
        "UpdateBehavior": "UPDATE_IN_DATABASE",
        "DeleteBehavior": "LOG",
    },
)

# Start the crawler
glue.start_crawler(Name="titanic-raw-crawler")

# Wait for completion
import time
while True:
    response = glue.get_crawler(Name="titanic-raw-crawler")
    state = response["Crawler"]["State"]
    if state == "READY":
        break
    time.sleep(10)

# Inspect cataloged table
table = glue.get_table(DatabaseName="ml_pipeline_db", Name="raw_titanic")
columns = table["Table"]["StorageDescriptor"]["Columns"]
print(f"Cataloged {len(columns)} columns: {[c['Name'] for c in columns]}")
```

#### 4.3 Glue ETL Job — Clean and Transform Data
```python
# Glue ETL script (runs inside Glue job environment)
# File: glue_scripts/etl_clean_titanic.py

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.transforms import *
from pyspark.context import SparkContext
from pyspark.sql import functions as F

sc = SparkContext()
glue_context = GlueContext(sc)
spark = glue_context.spark_session
job = Job(glue_context)

# Read from Glue Data Catalog
dyf = glue_context.create_dynamic_frame.from_catalog(
    database="ml_pipeline_db",
    table_name="raw_titanic",
)

# Convert to Spark DataFrame for transformations
df = dyf.toDF()

# --- Cleaning ---
# Drop duplicates
df = df.dropDuplicates()

# Fill nulls
df = df.fillna({"Age": df.agg(F.median("Age")).first()[0]})
df = df.fillna({"Fare": 0.0})

# Standardize string columns
df = df.withColumn("Sex", F.lower(F.trim(F.col("Sex"))))
df = df.withColumn("Name", F.trim(F.col("Name")))

# Add derived columns
df = df.withColumn("Age_Group",
    F.when(F.col("Age") < 12, "Child")
     .when(F.col("Age") < 18, "Teen")
     .when(F.col("Age") < 35, "Adult")
     .when(F.col("Age") < 60, "Middle")
     .otherwise("Senior")
)

# --- Write cleaned data back to S3 as Parquet ---
df.write.mode("overwrite").parquet(f"s3://{bucket}/cleaned/titanic/")

job.commit()
```

**Submit the Glue ETL job via boto3**:
```python
glue.create_job(
    Name="etl-clean-titanic",
    Role="arn:aws:iam::123456789012:role/GlueServiceRole",
    Command={
        "Name": "glueetl",
        "ScriptLocation": f"s3://{bucket}/glue_scripts/etl_clean_titanic.py",
        "PythonVersion": "3",
    },
    GlueVersion="4.0",
    NumberOfWorkers=2,
    WorkerType="G.1X",
)

# Run the job
run = glue.start_job_run(Name="etl-clean-titanic")
job_run_id = run["JobRunId"]

# Poll until complete
while True:
    status = glue.get_job_run(JobName="etl-clean-titanic", RunId=job_run_id)
    state = status["JobRun"]["JobRunState"]
    if state in ("SUCCEEDED", "FAILED", "STOPPED"):
        print(f"Glue job finished: {state}")
        break
    time.sleep(30)
```

#### 4.4 Re-Crawl Cleaned Data
```python
# Create crawler for cleaned output
glue.create_crawler(
    Name="titanic-cleaned-crawler",
    Role="arn:aws:iam::123456789012:role/GlueServiceRole",
    DatabaseName="ml_pipeline_db",
    Targets={"S3Targets": [{"Path": f"s3://{bucket}/cleaned/titanic/"}]},
    TablePrefix="cleaned_",
)

glue.start_crawler(Name="titanic-cleaned-crawler")
```

#### 4.5 Athena — Query Cleaned Data with SQL
```python
import time
import pandas as pd

athena = boto3.client("athena")

def run_athena_query(query: str, database: str, output_location: str) -> pd.DataFrame:
    """Execute an Athena query and return results as a DataFrame."""
    execution = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation": output_location},
    )
    execution_id = execution["QueryExecutionId"]

    # Wait for query to complete
    while True:
        result = athena.get_query_execution(QueryExecutionId=execution_id)
        state = result["QueryExecution"]["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(2)

    if state != "SUCCEEDED":
        reason = result["QueryExecution"]["Status"].get("StateChangeReason", "Unknown")
        raise RuntimeError(f"Athena query {state}: {reason}")

    # Read results from S3
    output_key = f"{output_location.split('/', 3)[3]}/{execution_id}.csv"
    return pd.read_csv(f"{output_location}/{execution_id}.csv")


# --- Example Queries ---

# Basic SELECT
df = run_athena_query(
    query="SELECT * FROM cleaned_titanic LIMIT 100",
    database="ml_pipeline_db",
    output_location=f"s3://{bucket}/athena-results/",
)

# Aggregation query
survival_by_class = run_athena_query(
    query="""
        SELECT pclass,
               COUNT(*) AS total,
               SUM(survived) AS survived,
               ROUND(AVG(survived) * 100, 1) AS survival_rate_pct
        FROM cleaned_titanic
        GROUP BY pclass
        ORDER BY pclass
    """,
    database="ml_pipeline_db",
    output_location=f"s3://{bucket}/athena-results/",
)

# CTAS — Create a new table from query (materialized view)
run_athena_query(
    query="""
        CREATE TABLE ml_pipeline_db.titanic_features
        WITH (format = 'PARQUET', external_location = 's3://bucket/features/titanic/')
        AS SELECT survived, pclass, sex, age, fare, age_group,
                  fare / NULLIF(siblings_spouses_aboard + 1, 0) AS fare_per_person
        FROM cleaned_titanic
        WHERE age IS NOT NULL
    """,
    database="ml_pipeline_db",
    output_location=f"s3://{bucket}/athena-results/",
)
```

#### 4.6 Download Athena Results for ML Pipeline
```python
# Pull Athena query results directly into pandas for downstream ML
features_df = run_athena_query(
    query="SELECT * FROM titanic_features",
    database="ml_pipeline_db",
    output_location=f"s3://{bucket}/athena-results/",
)

# Save locally for the data engineering stage
features_df.to_csv("data/extracted/titanic_from_athena.csv", index=False)
features_df.to_parquet("data/extracted/titanic_from_athena.parquet", index=False)
```

#### 4.7 Downstream Export to Other Data Stores

**Export to RDS (PostgreSQL/MySQL)**:
```python
from sqlalchemy import create_engine

engine = create_engine("postgresql://user:pass@rds-host:5432/mldb")
features_df.to_sql("titanic_features", engine, if_exists="replace", index=False)
```

**Export to DynamoDB**:
```python
dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table("titanic_predictions")

with table.batch_writer() as batch:
    for _, row in features_df.iterrows():
        batch.put_item(Item={
            "passenger_id": int(row.name),
            "survived": int(row["survived"]),
            "pclass": int(row["pclass"]),
            "age_group": row["age_group"],
        })
```

**Export to Redshift via S3 COPY** (Redshift infra already provisioned):
```python
# Write to S3 in Redshift-friendly format
features_df.to_csv(f"s3://{bucket}/redshift-staging/titanic_features.csv",
                    index=False)

# Redshift COPY command (run via psycopg2 or redshift-connector)
COPY_SQL = f"""
    COPY ml_schema.titanic_features
    FROM 's3://{bucket}/redshift-staging/titanic_features.csv'
    IAM_ROLE 'arn:aws:iam::123456789012:role/RedshiftCopyRole'
    CSV IGNOREHEADER 1;
"""
```

**Export back to S3 as Parquet for SageMaker**:
```python
import pyarrow as pa
import pyarrow.parquet as pq

table = pa.Table.from_pandas(features_df)
pq.write_table(table, f"s3://{bucket}/ml-ready/titanic/features.parquet")
```

### Data Flow Summary

```
┌──────────────┐     ┌───────────┐     ┌───────────────┐     ┌──────────────┐
│  Raw Files   │────▶│  S3 Raw   │────▶│ Glue Crawler  │────▶│ Glue Catalog │
│  (CSV/JSON/  │     │  Layer    │     │ (auto-schema) │     │ (database +  │
│   Parquet)   │     │           │     │               │     │   tables)    │
└──────────────┘     └───────────┘     └───────────────┘     └──────┬───────┘
                                                                    │
                     ┌───────────┐     ┌───────────────┐            │
                     │ S3 Clean  │◀────│ Glue ETL Job  │◀───────────┘
                     │ Layer     │     │ (PySpark)     │
                     └─────┬─────┘     └───────────────┘
                           │
              ┌────────────┼─────────────┐
              ▼            ▼             ▼
       ┌──────────┐ ┌───────────┐ ┌──────────────┐
       │  Athena   │ │ SageMaker │ │  Downstream  │
       │  Queries  │ │ Training  │ │  (RDS/Dynamo │
       │  (SQL)    │ │ (ML)      │ │  /Redshift)  │
       └──────────┘ └───────────┘ └──────────────┘
```

---

## 5. Stage 2 — Data Engineering

**Script**: `code_folder/22_data_engineering_pipeline.py`
**Input**: Raw data from Stage 1 or cleaned data from Stage 1B (Athena/S3)

### Techniques Covered

#### 5.1 Handle Missing Values
```python
# Numeric: fill with median
numeric_cols = df.select_dtypes(include="number").columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Categorical: fill with mode
cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
```

#### 5.2 Type Casting and Cleanup
```python
df["Age"] = df["Age"].astype(float)
df["Pclass"] = df["Pclass"].astype("category")

# Strip whitespace from string columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()
```

#### 5.3 Feature Engineering
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

#### 5.4 Encoding Categorical Variables
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Label encoding for binary columns
le = LabelEncoder()
df["Sex_Encoded"] = le.fit_transform(df["Sex"])

# One-hot encoding for multi-class columns
df = pd.get_dummies(df, columns=["Age_Group"], prefix="Age", drop_first=True)
```

#### 5.5 Normalization / Scaling
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()
df[["Fare_Scaled"]] = scaler.fit_transform(df[["Fare"]])
```

#### 5.6 Train / Test Split with Stratification
```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

---

## 6. Stage 3 — ML Model Training

**Script**: `code_folder/22_ml_model_training.py`

### Techniques Covered

#### 6.1 Baseline Model (Logistic Regression)
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_train)
print(f"Train Accuracy: {accuracy_score(y_train, y_pred):.4f}")
```

#### 6.2 Gradient Boosted Model (XGBoost)
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

#### 6.3 Cross-Validation
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

#### 6.4 Save Model Artifacts
```python
import joblib

joblib.dump(model, "model_artifacts/logistic_model.joblib")
joblib.dump(xgb_model, "model_artifacts/xgb_model.joblib")
joblib.dump(scaler, "model_artifacts/scaler.joblib")
joblib.dump(le, "model_artifacts/label_encoder.joblib")
```

---

## 7. Stage 4 — Inference & Overlay on Test Dataset

**Script**: `code_folder/22_ml_inference_overlay.py`

### Techniques Covered

#### 7.1 Load Model and Run Inference
```python
import joblib
import pandas as pd

model = joblib.load("model_artifacts/xgb_model.joblib")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
```

#### 7.2 Overlay Predictions on Test DataFrame
```python
test_results = X_test.copy()
test_results["Actual"] = y_test.values
test_results["Predicted"] = y_pred
test_results["Predicted_Probability"] = y_proba
test_results["Correct"] = test_results["Actual"] == test_results["Predicted"]
```

#### 7.3 Evaluate on Test Set
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

#### 7.4 Export Overlayed Results
```python
test_results.to_csv("output/test_predictions_overlay.csv", index=False)
test_results.to_parquet("output/test_predictions_overlay.parquet", index=False)

pd.DataFrame([metrics]).to_csv("output/test_metrics.csv", index=False)
```

#### 7.5 Error Analysis
```python
errors = test_results[~test_results["Correct"]]
print(f"Total errors: {len(errors)} / {len(test_results)}")
print(f"Error rate: {len(errors) / len(test_results):.2%}")
print(f"\nError distribution by class:\n{errors['Actual'].value_counts()}")
```

---

## 8. Running in SageMaker

> Infrastructure (SageMaker domain, IAM roles, S3 buckets) is already provisioned. The instructions below cover **how to run the pipeline code** inside SageMaker.

### 8.1 SageMaker Studio Notebook (Interactive)

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
%run code_folder/22_glue_s3_athena_pipeline.py
%run code_folder/22_data_engineering_pipeline.py
%run code_folder/22_ml_model_training.py
%run code_folder/22_ml_inference_overlay.py
```

### 8.2 SageMaker Processing Jobs (Batch Execution)

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

### 8.3 Glue Integration in SageMaker Pipelines

Trigger Glue Crawlers and ETL jobs from within SageMaker Pipelines using callback steps or Lambda steps.

**Option A — SageMaker Pipeline CallbackStep with Glue**:
```python
from sagemaker.workflow.callback_step import CallbackStep
from sagemaker.workflow.lambda_step import LambdaStep, Lambda

# Lambda function that triggers Glue crawler + ETL job
glue_lambda = Lambda(
    function_name="trigger-glue-etl",
    execution_role_arn=role,
    script="lambda_functions/trigger_glue.py",
)

glue_step = LambdaStep(
    name="GlueETL",
    lambda_func=glue_lambda,
    inputs={"crawler_name": "titanic-raw-crawler", "job_name": "etl-clean-titanic"},
    outputs=["cleaned_s3_path"],
)
```

**Option B — Glue Job triggered from Processing Job**:
```python
# In 22_glue_s3_athena_pipeline.py, the script uses boto3 to:
# 1. Upload data to S3
# 2. Start Glue Crawler and wait
# 3. Start Glue ETL Job and wait
# 4. Run Athena queries on cleaned data
# 5. Export results for downstream stages
```

**Option C — Athena query from SageMaker Processing Job**:
```python
sklearn_processor.run(
    code="code_folder/22_glue_s3_athena_pipeline.py",
    outputs=[
        ProcessingOutput(
            output_name="athena_results",
            source="/opt/ml/processing/output",
            destination=f"s3://{bucket}/{prefix}/athena-output/",
        )
    ],
)
```

### 8.4 SageMaker Training Jobs

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

### 8.5 SageMaker Batch Transform (Inference at Scale)

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

### 8.6 SageMaker Pipelines (Full Orchestration)

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

### 8.7 SageMaker Experiments (Tracking)

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
| Glue ETL / Crawlers | Glue Job (via LambdaStep or boto3) | Glue G.1X (2-10 workers) |
| Athena Queries | Processing Job (boto3 + Athena) | ml.m5.xlarge |
| Data Engineering | Processing Job (SKLearnProcessor) | ml.m5.xlarge |
| Model Training | Training Job (SKLearn Estimator) | ml.m5.xlarge (CPU) or ml.g4dn.xlarge (GPU) |
| Inference / Overlay | Batch Transform | ml.m5.xlarge |
| Downstream Export | Processing Job or Lambda | ml.m5.large |
| Full Pipeline | SageMaker Pipelines | Per-step config |
| Experiment Tracking | SageMaker Experiments | N/A (metadata only) |

---

## 9. Exercise List

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
| 11 | Upload CSV/Parquet to S3, verify with `list_objects_v2` | `boto3 s3.upload_file()`, S3 keys |
| 12 | Run a Glue Crawler on S3 data, inspect the cataloged table | `glue.start_crawler()`, `get_table()` |
| 13 | Query S3 data via Athena, return results as DataFrame | `athena.start_query_execution()`, SQL |
| 14 | Build a feature engineering pipeline with `ColumnTransformer` | `Pipeline`, `ColumnTransformer` |
| 15 | Train XGBoost with cross-validation, report mean CV score | `XGBClassifier`, `cross_val_score()` |
| 16 | Compare Logistic Regression vs XGBoost on same test set | Side-by-side metrics, overlay both |
| 17 | Save and reload model artifacts with joblib | `joblib.dump()`, `joblib.load()` |
| 18 | Compute confusion matrix and classification report | `confusion_matrix()`, `classification_report()` |
| 19 | Run data extraction as a SageMaker Processing Job | `SKLearnProcessor`, `ProcessingOutput` |

### Advanced

| # | Exercise | Key Concepts |
|---|----------|-------------|
| 20 | Build end-to-end pipeline: extract → Glue → engineer → train → infer | Multi-script orchestration |
| 21 | Write a Glue ETL job (PySpark) to clean and transform S3 data | `GlueContext`, `DynamicFrame`, PySpark |
| 22 | Create partitioned S3 data, crawl it, query with Athena partition pruning | S3 partitioning, `MSCK REPAIR TABLE` |
| 23 | Use Athena CTAS to materialize a feature table from raw data | `CREATE TABLE AS SELECT`, Parquet output |
| 24 | Export Athena results to RDS and DynamoDB downstream | `sqlalchemy`, `dynamodb.batch_writer()` |
| 25 | Extract from multiple sources, merge into single dataset | `pd.merge()`, API + CSV join |
| 26 | Implement data validation checks between stages | Schema checks, row count assertions |
| 27 | Overlay predictions with confidence intervals | `predict_proba()`, threshold analysis |
| 28 | Error analysis: segment misclassifications by feature group | Groupby on error dataframe |
| 29 | Orchestrate full pipeline with SageMaker Pipelines + Glue | `Pipeline`, `LambdaStep`, Glue triggers |
| 30 | Track experiments with SageMaker Experiments | `Run`, `log_metric()`, `log_parameter()` |
| 31 | Run batch inference with SageMaker Batch Transform | `Transformer`, `transform()` |
| 32 | Write Athena query results back to S3 as Parquet for Redshift COPY | Athena → S3 → Redshift COPY |

---

## 10. References

- [AWS Glue Developer Guide](https://docs.aws.amazon.com/glue/latest/dg/what-is-glue.html)
- [AWS Glue PySpark Extensions](https://docs.aws.amazon.com/glue/latest/dg/aws-glue-programming-python.html)
- [Amazon Athena User Guide](https://docs.aws.amazon.com/athena/latest/ug/what-is.html)
- [Athena SQL Reference](https://docs.aws.amazon.com/athena/latest/ug/ddl-sql-reference.html)
- [boto3 Glue Client](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/glue.html)
- [boto3 Athena Client](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/athena.html)
- [SageMaker Python SDK — Processing](https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_processing.html)
- [SageMaker Python SDK — Training](https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/sagemaker.sklearn.html)
- [SageMaker Python SDK — Pipelines](https://sagemaker.readthedocs.io/en/stable/workflows/pipelines/index.html)
- [SageMaker Python SDK — Experiments](https://sagemaker.readthedocs.io/en/stable/experiments/sagemaker.experiments.html)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [UCI ML Repository](https://archive.ics.uci.edu/)
