"""
Category 22: Glue ETL, S3 Data Lake & Athena Pipeline
=======================================================
Demonstrates the AWS data lake pattern:
  1. Upload raw data to S3
  2. Run Glue Crawlers to catalog schema
  3. Run Glue ETL jobs to clean/transform
  4. Query cleaned data with Athena
  5. Export results downstream (RDS, DynamoDB, S3 for Redshift)

Prerequisites:
  - AWS credentials configured (boto3 access)
  - Glue service role, S3 bucket, and Glue database already provisioned
  - Athena workgroup and results bucket configured

Usage:
    python 22_glue_s3_athena_pipeline.py

    # With custom bucket/database:
    python 22_glue_s3_athena_pipeline.py \\
        --bucket my-data-lake \\
        --database ml_pipeline_db \\
        --glue-role arn:aws:iam::123456789012:role/GlueServiceRole

SageMaker Processing Job:
    Input:  /opt/ml/processing/input/   (extracted CSV/Parquet files)
    Output: /opt/ml/processing/output/  (Athena query results)
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CLI Arguments
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", type=str, default="ml-pipeline-data-lake")
    parser.add_argument("--prefix", type=str, default="ml-pipeline-practice")
    parser.add_argument("--database", type=str, default="ml_pipeline_db")
    parser.add_argument("--glue-role", type=str, default="")
    parser.add_argument("--athena-output", type=str, default="")
    parser.add_argument("--mode", type=str, default="local",
                        choices=["local", "aws"],
                        help="'local' simulates with local files; 'aws' uses real AWS services")
    args, _ = parser.parse_known_args()
    return args


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def get_input_dir() -> Path:
    if os.path.isdir("/opt/ml/processing/input"):
        return Path("/opt/ml/processing/input")
    return Path(__file__).parent.parent / "data" / "extracted"


def get_output_dir() -> Path:
    if os.path.isdir("/opt/ml"):
        output_dir = Path("/opt/ml/processing/output")
    else:
        output_dir = Path(__file__).parent.parent / "data" / "glue_athena_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ==========================================================================
# S3 Operations
# ==========================================================================
class S3DataLake:
    """Upload, list, and manage data in S3."""

    def __init__(self, bucket: str, prefix: str):
        import boto3
        self.s3 = boto3.client("s3")
        self.bucket = bucket
        self.prefix = prefix

    def upload_file(self, local_path: str, s3_key: str) -> str:
        """Upload a local file to S3."""
        full_key = f"{self.prefix}/{s3_key}"
        print(f"[S3] Uploading {local_path} -> s3://{self.bucket}/{full_key}")
        self.s3.upload_file(local_path, self.bucket, full_key)
        return f"s3://{self.bucket}/{full_key}"

    def upload_dataframe_csv(self, df: pd.DataFrame, s3_key: str) -> str:
        """Upload a DataFrame as CSV to S3."""
        full_key = f"{self.prefix}/{s3_key}"
        csv_buffer = df.to_csv(index=False)
        print(f"[S3] Uploading DataFrame ({df.shape[0]} rows) -> s3://{self.bucket}/{full_key}")
        self.s3.put_object(Bucket=self.bucket, Key=full_key, Body=csv_buffer.encode())
        return f"s3://{self.bucket}/{full_key}"

    def upload_dataframe_parquet(self, df: pd.DataFrame, s3_key: str) -> str:
        """Upload a DataFrame as Parquet to S3."""
        full_key = f"{self.prefix}/{s3_key}"
        parquet_buffer = df.to_parquet(index=False)
        print(f"[S3] Uploading DataFrame ({df.shape[0]} rows) -> s3://{self.bucket}/{full_key}")
        self.s3.put_object(Bucket=self.bucket, Key=full_key, Body=parquet_buffer)
        return f"s3://{self.bucket}/{full_key}"

    def list_objects(self, prefix_suffix: str = "") -> list[str]:
        """List objects under the configured prefix."""
        full_prefix = f"{self.prefix}/{prefix_suffix}" if prefix_suffix else self.prefix
        response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=full_prefix)
        keys = [obj["Key"] for obj in response.get("Contents", [])]
        print(f"[S3] Listed {len(keys)} objects under s3://{self.bucket}/{full_prefix}")
        return keys

    def download_to_dataframe(self, s3_key: str, file_format: str = "csv") -> pd.DataFrame:
        """Download an S3 object as a DataFrame."""
        full_key = f"{self.prefix}/{s3_key}"
        s3_uri = f"s3://{self.bucket}/{full_key}"
        print(f"[S3] Downloading {s3_uri}")
        if file_format == "parquet":
            return pd.read_parquet(s3_uri)
        return pd.read_csv(s3_uri)


# ==========================================================================
# Glue Operations
# ==========================================================================
class GlueCatalog:
    """Manage Glue Crawlers, ETL Jobs, and Data Catalog."""

    def __init__(self, database: str, role_arn: str):
        import boto3
        self.glue = boto3.client("glue")
        self.database = database
        self.role_arn = role_arn

    # --- Database ---
    def ensure_database(self):
        """Create Glue database if it doesn't exist."""
        try:
            self.glue.get_database(Name=self.database)
            print(f"[Glue] Database '{self.database}' exists")
        except self.glue.exceptions.EntityNotFoundException:
            self.glue.create_database(
                DatabaseInput={"Name": self.database, "Description": "ML pipeline data catalog"}
            )
            print(f"[Glue] Created database '{self.database}'")

    # --- Crawler ---
    def create_and_run_crawler(self, name: str, s3_path: str, table_prefix: str = "") -> dict:
        """Create a Glue Crawler, run it, and return cataloged table info."""
        # Create or update crawler
        crawler_config = {
            "Name": name,
            "Role": self.role_arn,
            "DatabaseName": self.database,
            "Targets": {"S3Targets": [{"Path": s3_path}]},
            "TablePrefix": table_prefix,
            "SchemaChangePolicy": {
                "UpdateBehavior": "UPDATE_IN_DATABASE",
                "DeleteBehavior": "LOG",
            },
        }
        try:
            self.glue.get_crawler(Name=name)
            self.glue.update_crawler(**crawler_config)
            print(f"[Glue] Updated crawler '{name}'")
        except self.glue.exceptions.EntityNotFoundException:
            self.glue.create_crawler(**crawler_config)
            print(f"[Glue] Created crawler '{name}'")

        # Start crawler
        print(f"[Glue] Starting crawler '{name}' on {s3_path}")
        self.glue.start_crawler(Name=name)

        # Wait for completion
        while True:
            response = self.glue.get_crawler(Name=name)
            state = response["Crawler"]["State"]
            if state == "READY":
                last_crawl = response["Crawler"].get("LastCrawl", {})
                status = last_crawl.get("Status", "UNKNOWN")
                print(f"[Glue] Crawler '{name}' finished: {status}")
                if last_crawl.get("ErrorMessage"):
                    print(f"  -> Error: {last_crawl['ErrorMessage']}")
                return last_crawl
            print(f"  -> Crawler state: {state}, waiting...")
            time.sleep(15)

    # --- ETL Job ---
    def create_and_run_etl_job(
        self, name: str, script_s3_path: str, args: dict = None
    ) -> str:
        """Create and run a Glue ETL job."""
        # Create or update job
        job_config = {
            "Name": name,
            "Role": self.role_arn,
            "Command": {
                "Name": "glueetl",
                "ScriptLocation": script_s3_path,
                "PythonVersion": "3",
            },
            "GlueVersion": "4.0",
            "NumberOfWorkers": 2,
            "WorkerType": "G.1X",
            "DefaultArguments": {
                "--enable-metrics": "true",
                "--enable-continuous-cloudwatch-log": "true",
                **(args or {}),
            },
        }
        try:
            self.glue.get_job(JobName=name)
            self.glue.update_job(JobName=name, JobUpdate=job_config)
            print(f"[Glue] Updated ETL job '{name}'")
        except self.glue.exceptions.EntityNotFoundException:
            self.glue.create_job(**job_config)
            print(f"[Glue] Created ETL job '{name}'")

        # Run job
        run_args = args or {}
        run = self.glue.start_job_run(JobName=name, Arguments=run_args)
        run_id = run["JobRunId"]
        print(f"[Glue] Started ETL job '{name}' (run: {run_id})")

        # Wait for completion
        while True:
            status = self.glue.get_job_run(JobName=name, RunId=run_id)
            state = status["JobRun"]["JobRunState"]
            if state in ("SUCCEEDED", "FAILED", "STOPPED", "TIMEOUT"):
                print(f"[Glue] ETL job '{name}' finished: {state}")
                if state != "SUCCEEDED":
                    msg = status["JobRun"].get("ErrorMessage", "")
                    print(f"  -> Error: {msg}")
                return state
            print(f"  -> Job state: {state}, waiting...")
            time.sleep(30)

    # --- Catalog inspection ---
    def get_table_info(self, table_name: str) -> dict:
        """Get schema info for a cataloged table."""
        table = self.glue.get_table(DatabaseName=self.database, Name=table_name)
        columns = table["Table"]["StorageDescriptor"]["Columns"]
        location = table["Table"]["StorageDescriptor"]["Location"]
        info = {
            "table_name": table_name,
            "location": location,
            "columns": [{"name": c["Name"], "type": c["Type"]} for c in columns],
            "num_columns": len(columns),
        }
        print(f"[Glue] Table '{table_name}': {len(columns)} columns at {location}")
        return info

    def list_tables(self) -> list[str]:
        """List all tables in the database."""
        response = self.glue.get_tables(DatabaseName=self.database)
        tables = [t["Name"] for t in response["TableList"]]
        print(f"[Glue] Database '{self.database}' has {len(tables)} tables: {tables}")
        return tables


# ==========================================================================
# Athena Operations
# ==========================================================================
class AthenaQuery:
    """Run SQL queries via Athena on Glue-cataloged data."""

    def __init__(self, database: str, output_location: str):
        import boto3
        self.athena = boto3.client("athena")
        self.database = database
        self.output_location = output_location

    def run_query(self, query: str, description: str = "") -> pd.DataFrame:
        """Execute an Athena query and return results as DataFrame."""
        if description:
            print(f"[Athena] {description}")
        print(f"[Athena] Executing: {query[:120]}...")

        execution = self.athena.start_query_execution(
            QueryString=query,
            QueryExecutionContext={"Database": self.database},
            ResultConfiguration={"OutputLocation": self.output_location},
        )
        execution_id = execution["QueryExecutionId"]

        # Wait for completion
        while True:
            result = self.athena.get_query_execution(QueryExecutionId=execution_id)
            state = result["QueryExecution"]["Status"]["State"]
            if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
                break
            time.sleep(2)

        if state != "SUCCEEDED":
            reason = result["QueryExecution"]["Status"].get("StateChangeReason", "Unknown")
            raise RuntimeError(f"Athena query {state}: {reason}")

        # Get results
        output_uri = result["QueryExecution"]["ResultConfiguration"]["OutputLocation"]
        print(f"[Athena] Query succeeded. Results at {output_uri}")

        # Paginate results into DataFrame
        rows = []
        columns = []
        paginator = self.athena.get_paginator("get_query_results")
        for page in paginator.paginate(QueryExecutionId=execution_id):
            result_set = page["ResultSet"]
            if not columns:
                columns = [col["Label"] for col in result_set["ResultSetMetadata"]["ColumnInfo"]]
            for row in result_set["Rows"][1 if not rows else 0:]:  # skip header on first page
                rows.append([field.get("VarCharValue", None) for field in row["Data"]])

        df = pd.DataFrame(rows, columns=columns)
        print(f"[Athena] Returned {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def create_table_as_select(self, new_table: str, query: str, s3_location: str) -> pd.DataFrame:
        """Create a new Glue table from a SELECT query (CTAS)."""
        ctas = f"""
            CREATE TABLE {self.database}.{new_table}
            WITH (format = 'PARQUET', external_location = '{s3_location}')
            AS {query}
        """
        return self.run_query(ctas, description=f"CTAS -> {new_table}")


# ==========================================================================
# Downstream Export
# ==========================================================================
class DownstreamExporter:
    """Export query results to other data stores."""

    @staticmethod
    def to_rds_postgres(df: pd.DataFrame, table_name: str, connection_string: str):
        """Export DataFrame to RDS PostgreSQL."""
        from sqlalchemy import create_engine
        engine = create_engine(connection_string)
        df.to_sql(table_name, engine, if_exists="replace", index=False, method="multi")
        print(f"[Export] Wrote {len(df)} rows to RDS table '{table_name}'")

    @staticmethod
    def to_dynamodb(df: pd.DataFrame, table_name: str, key_column: str):
        """Export DataFrame rows to DynamoDB.

        Args:
            df: DataFrame to export.
            table_name: DynamoDB table name.
            key_column: Column to use as the DynamoDB partition key.
                        Must exist in the DataFrame.
        """
        if key_column not in df.columns:
            raise ValueError(
                f"key_column '{key_column}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

        import boto3
        dynamodb = boto3.resource("dynamodb")
        table = dynamodb.Table(table_name)
        written = 0
        with table.batch_writer() as batch:
            for _, row in df.iterrows():
                item = {k: _convert_dynamodb_type(v) for k, v in row.to_dict().items()}
                if key_column not in item or item[key_column] is None:
                    raise ValueError(f"Partition key '{key_column}' is missing or None for row")
                batch.put_item(Item=item)
                written += 1
        print(f"[Export] Wrote {written} items to DynamoDB table '{table_name}' (key={key_column})")

    @staticmethod
    def to_s3_for_redshift(df: pd.DataFrame, bucket: str, key: str) -> str:
        """Write CSV to S3 for Redshift COPY command."""
        import boto3
        s3 = boto3.client("s3")
        csv_data = df.to_csv(index=False)
        s3.put_object(Bucket=bucket, Key=key, Body=csv_data.encode())
        s3_uri = f"s3://{bucket}/{key}"
        print(f"[Export] Wrote {len(df)} rows to {s3_uri} (ready for Redshift COPY)")
        print("  -> COPY command:")
        print(f"     COPY schema.table FROM '{s3_uri}' IAM_ROLE 'arn:...' CSV IGNOREHEADER 1;")
        return s3_uri


def _convert_dynamodb_type(value):
    """Convert Python/numpy types for DynamoDB compatibility."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return str(round(float(value), 6))
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


# ==========================================================================
# Local Simulation (runs without AWS)
# ==========================================================================
def run_local_simulation(input_dir: Path, output_dir: Path):
    """
    Simulate the Glue/S3/Athena pipeline locally.
    Useful for development and SageMaker Processing Jobs
    where Glue/Athena aren't directly available.
    """
    print("=" * 60)
    print("[Local Mode] Simulating Glue/S3/Athena pipeline locally")
    print("=" * 60)

    # --- 1. Load raw data (simulates S3 download) ---
    print("\n--- Step 1: Load raw data (simulates S3 ingestion) ---")
    csv_path = input_dir / "titanic_raw.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        print(f"  -> {csv_path} not found, using sklearn Titanic-like data")
        from sklearn.datasets import fetch_california_housing
        housing = fetch_california_housing(as_frame=True)
        df = housing.frame
    print(f"  -> Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    # --- 2. Simulate Glue Crawler (schema discovery) ---
    print("\n--- Step 2: Schema discovery (simulates Glue Crawler) ---")
    schema = {
        col: str(dtype)
        for col, dtype in df.dtypes.items()
    }
    print(f"  -> Discovered {len(schema)} columns:")
    for col, dtype in schema.items():
        print(f"     {col}: {dtype}")

    # --- 3. Simulate Glue ETL (clean + transform) ---
    print("\n--- Step 3: Clean & transform (simulates Glue ETL job) ---")
    df_cleaned = df.copy()

    # Drop duplicates
    before = len(df_cleaned)
    df_cleaned = df_cleaned.drop_duplicates()
    print(f"  -> Dropped {before - len(df_cleaned)} duplicates")

    # Fill nulls
    for col in df_cleaned.select_dtypes(include="number").columns:
        null_count = df_cleaned[col].isnull().sum()
        if null_count > 0:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
            print(f"  -> Filled {null_count} nulls in '{col}' with median")

    for col in df_cleaned.select_dtypes(include="object").columns:
        null_count = df_cleaned[col].isnull().sum()
        if null_count > 0:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
            print(f"  -> Filled {null_count} nulls in '{col}' with mode")

    # Standardize strings
    for col in df_cleaned.select_dtypes(include="object").columns:
        df_cleaned[col] = df_cleaned[col].str.strip().str.lower()
        print(f"  -> Standardized strings in '{col}'")

    # Save cleaned data (simulates writing back to S3)
    cleaned_csv = output_dir / "cleaned_data.csv"
    cleaned_parquet = output_dir / "cleaned_data.parquet"
    df_cleaned.to_csv(cleaned_csv, index=False)
    df_cleaned.to_parquet(cleaned_parquet, index=False)
    print(f"  -> Saved cleaned data: {df_cleaned.shape}")

    # --- 4. Simulate Athena queries ---
    print("\n--- Step 4: SQL queries (simulates Athena) ---")

    # Use DuckDB as a local Athena stand-in
    try:
        import duckdb

        con = duckdb.connect()
        con.register("cleaned_data", df_cleaned)

        # Basic SELECT
        sample = con.execute("SELECT * FROM cleaned_data LIMIT 5").fetchdf()
        print(f"  -> SELECT * LIMIT 5: {sample.shape}")

        # Aggregation (generic — works for any dataset)
        numeric_cols = df_cleaned.select_dtypes(include="number").columns.tolist()
        if len(numeric_cols) >= 2:
            target_col = numeric_cols[0]
            group_col = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
            agg_query = f"""
                SELECT ROUND({group_col}, 0) AS group_val,
                       COUNT(*) AS cnt,
                       AVG({target_col}) AS avg_target
                FROM cleaned_data
                GROUP BY ROUND({group_col}, 0)
                ORDER BY cnt DESC
                LIMIT 10
            """
            agg_result = con.execute(agg_query).fetchdf()
            print(f"  -> Aggregation query: {agg_result.shape[0]} groups")
            agg_result.to_csv(output_dir / "athena_aggregation.csv", index=False)

        # Full export (simulates Athena CTAS)
        full_result = con.execute("SELECT * FROM cleaned_data").fetchdf()
        full_result.to_parquet(output_dir / "athena_full_export.parquet", index=False)
        print(f"  -> Full export: {full_result.shape}")

        con.close()

    except ImportError:
        print("  -> DuckDB not installed, using pandas for SQL simulation")
        df_cleaned.to_parquet(output_dir / "athena_full_export.parquet", index=False)

    # --- 5. Simulate downstream export ---
    print("\n--- Step 5: Downstream export ---")

    # Export as CSV for Redshift COPY
    redshift_path = output_dir / "redshift_staging.csv"
    df_cleaned.to_csv(redshift_path, index=False)
    print(f"  -> Redshift staging CSV: {redshift_path}")

    # Export as JSON for DynamoDB
    dynamodb_path = output_dir / "dynamodb_items.json"
    records = df_cleaned.head(100).to_dict(orient="records")
    with open(dynamodb_path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"  -> DynamoDB items JSON (first 100): {dynamodb_path}")

    # --- Metadata ---
    metadata = {
        "mode": "local_simulation",
        "input_rows": int(df.shape[0]),
        "cleaned_rows": int(df_cleaned.shape[0]),
        "columns": list(df_cleaned.columns),
        "schema": schema,
        "output_files": [
            str(cleaned_csv),
            str(cleaned_parquet),
            str(redshift_path),
            str(dynamodb_path),
        ],
    }
    with open(output_dir / "glue_athena_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[Done] Local simulation complete. Output: {output_dir}")
    return df_cleaned


# ==========================================================================
# AWS Mode (runs with real Glue/S3/Athena)
# ==========================================================================
def run_aws_pipeline(args, input_dir: Path, output_dir: Path):
    """Run the full Glue/S3/Athena pipeline on AWS."""
    print("=" * 60)
    print("[AWS Mode] Running Glue/S3/Athena pipeline")
    print("=" * 60)

    bucket = args.bucket
    prefix = args.prefix
    database = args.database
    glue_role = args.glue_role
    athena_output = args.athena_output or f"s3://{bucket}/{prefix}/athena-results/"

    # Initialize clients
    s3_lake = S3DataLake(bucket, prefix)
    glue = GlueCatalog(database, glue_role)
    athena = AthenaQuery(database, athena_output)

    # --- 1. Upload raw data to S3 ---
    print("\n--- Step 1: Upload raw data to S3 ---")
    raw_files = list(input_dir.glob("*.csv")) + list(input_dir.glob("*.parquet"))
    for f in raw_files:
        ext = f.suffix
        s3_key = f"raw/{f.stem}/{f.name}"
        if ext == ".parquet":
            df = pd.read_parquet(f)
            s3_lake.upload_dataframe_parquet(df, s3_key)
        else:
            s3_lake.upload_file(str(f), s3_key)

    # --- 2. Ensure Glue database exists ---
    glue.ensure_database()

    # --- 3. Run Glue Crawler on raw data ---
    print("\n--- Step 2: Run Glue Crawlers ---")
    for f in raw_files:
        crawler_name = f"crawler-{f.stem}"
        s3_path = f"s3://{bucket}/{prefix}/raw/{f.stem}/"
        glue.create_and_run_crawler(crawler_name, s3_path, table_prefix="raw_")

    # --- 4. Inspect cataloged tables ---
    print("\n--- Step 3: Inspect catalog ---")
    tables = glue.list_tables()
    for table in tables:
        glue.get_table_info(table)

    # --- 5. Run Athena queries ---
    print("\n--- Step 4: Athena queries ---")
    for table in tables:
        # Basic query
        df_sample = athena.run_query(
            f"SELECT * FROM {table} LIMIT 100",
            description=f"Sample from {table}",
        )
        df_sample.to_csv(output_dir / f"athena_sample_{table}.csv", index=False)

    # --- 6. Download for ML pipeline ---
    print("\n--- Step 5: Export for ML pipeline ---")
    if tables:
        primary_table = tables[0]
        df_full = athena.run_query(f"SELECT * FROM {primary_table}")
        df_full.to_csv(output_dir / "athena_export.csv", index=False)
        df_full.to_parquet(output_dir / "athena_export.parquet", index=False)

    print(f"\n[Done] AWS pipeline complete. Output: {output_dir}")


# ==========================================================================
# Main
# ==========================================================================
def main():
    args = parse_args()
    input_dir = get_input_dir()
    output_dir = get_output_dir()

    if args.mode == "aws":
        run_aws_pipeline(args, input_dir, output_dir)
    else:
        run_local_simulation(input_dir, output_dir)


if __name__ == "__main__":
    main()
