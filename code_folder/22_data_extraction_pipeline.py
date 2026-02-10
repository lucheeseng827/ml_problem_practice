"""
Category 22: Data Extraction Pipeline
======================================
Extract data from multiple public sources:
- CSV from URL (Titanic dataset)
- sklearn built-in dataset (California Housing)
- REST API (World Bank GDP indicators)
- Parquet file (NYC Taxi - sampled)
- SQLite database (round-trip storage)

Usage:
    python 22_data_extraction_pipeline.py

SageMaker Processing Job:
    Output directory: /opt/ml/processing/output/
"""

import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.datasets import fetch_california_housing


def get_output_dir() -> Path:
    """Return output directory — uses SageMaker path if available."""
    sm_output = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/processing/output")
    if os.path.isdir("/opt/ml"):
        output_dir = Path(sm_output)
    else:
        output_dir = Path(__file__).parent.parent / "data" / "extracted"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ---------------------------------------------------------------------------
# 1. Extract CSV from URL — Titanic Dataset
# ---------------------------------------------------------------------------
def extract_titanic_csv() -> pd.DataFrame:
    """Download Titanic dataset from Stanford CS109."""
    url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
    print(f"[Extract] Downloading Titanic CSV from {url}")
    df = pd.read_csv(url)
    print(f"  -> {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  -> Columns: {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# 2. Extract sklearn Built-in Dataset — California Housing
# ---------------------------------------------------------------------------
def extract_california_housing() -> pd.DataFrame:
    """Load California Housing dataset from sklearn."""
    print("[Extract] Loading California Housing from sklearn")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    print(f"  -> {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  -> Target: MedHouseVal (median house value in $100k)")
    return df


# ---------------------------------------------------------------------------
# 3. Extract from REST API — World Bank GDP Data
# ---------------------------------------------------------------------------
def extract_world_bank_gdp(country: str = "US", start_year: int = 2000, end_year: int = 2023) -> pd.DataFrame:
    """Fetch GDP data from World Bank API."""
    url = f"https://api.worldbank.org/v2/country/{country}/indicator/NY.GDP.MKTP.CD"
    params = {
        "format": "json",
        "date": f"{start_year}:{end_year}",
        "per_page": 100,
    }
    print(f"[Extract] Fetching World Bank GDP for {country} ({start_year}-{end_year})")

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if len(data) < 2 or data[1] is None:
            print("  -> Warning: No data returned from API")
            return pd.DataFrame()

        records = data[1]
        df = pd.DataFrame(
            [
                {
                    "country": r["country"]["value"],
                    "country_code": r["countryiso3code"],
                    "year": int(r["date"]),
                    "gdp_usd": r["value"],
                    "indicator": r["indicator"]["value"],
                }
                for r in records
                if r["value"] is not None
            ]
        )
        print(f"  -> {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    except requests.RequestException as e:
        print(f"  -> API request failed: {e}")
        # Return synthetic fallback for offline/SageMaker environments
        print("  -> Generating synthetic GDP data as fallback")
        years = list(range(start_year, end_year + 1))
        rng = np.random.default_rng(42)
        return pd.DataFrame(
            {
                "country": ["United States"] * len(years),
                "country_code": ["USA"] * len(years),
                "year": years,
                "gdp_usd": rng.uniform(1e13, 2.5e13, size=len(years)),
                "indicator": ["GDP (current US$)"] * len(years),
            }
        )


# ---------------------------------------------------------------------------
# 4. Extract from Parquet — NYC Taxi (sampled)
# ---------------------------------------------------------------------------
def extract_nyc_taxi_sample(n_rows: int = 10000) -> pd.DataFrame:
    """
    Attempt to read NYC Taxi Parquet; fall back to synthetic data
    if the network call fails (common in restricted environments).
    """
    url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"
    print(f"[Extract] Attempting NYC Taxi parquet from {url}")

    try:
        df = pd.read_parquet(url)
        df = df.sample(n=min(n_rows, len(df)), random_state=42).reset_index(drop=True)
        print(f"  -> Sampled {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    except Exception as e:
        print(f"  -> Parquet download failed: {e}")
        print(f"  -> Generating synthetic taxi data ({n_rows} rows)")
        rng = np.random.default_rng(42)
        return pd.DataFrame(
            {
                "tpep_pickup_datetime": pd.date_range("2023-01-01", periods=n_rows, freq="2min"),
                "tpep_dropoff_datetime": pd.date_range("2023-01-01 00:15:00", periods=n_rows, freq="2min"),
                "passenger_count": rng.integers(1, 6, size=n_rows),
                "trip_distance": rng.exponential(3.0, size=n_rows).round(2),
                "fare_amount": rng.uniform(5, 80, size=n_rows).round(2),
                "tip_amount": rng.uniform(0, 20, size=n_rows).round(2),
                "total_amount": rng.uniform(8, 100, size=n_rows).round(2),
                "payment_type": rng.choice([1, 2, 3, 4], size=n_rows),
            }
        )


# ---------------------------------------------------------------------------
# 5. Store and Retrieve from SQLite
# ---------------------------------------------------------------------------
def round_trip_sqlite(df: pd.DataFrame, table_name: str, db_path: str) -> pd.DataFrame:
    """Write a dataframe to SQLite and read it back — simulates DB extraction."""
    print(f"[Extract] SQLite round-trip: writing {len(df)} rows to '{table_name}'")
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        df_back = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        print(f"  -> Read back {df_back.shape[0]} rows from SQLite")
        return df_back
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_extraction(df: pd.DataFrame, name: str) -> None:
    """Basic validation checks after extraction."""
    assert df.shape[0] > 0, f"{name}: No rows extracted"
    assert df.shape[1] > 0, f"{name}: No columns extracted"
    null_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
    print(f"[Validate] {name}: {df.shape[0]} rows, {df.shape[1]} cols, {null_pct:.1f}% null")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    output_dir = get_output_dir()
    db_path = str(output_dir / "extracted.db")

    # Extract from all sources
    titanic_df = extract_titanic_csv()
    housing_df = extract_california_housing()
    gdp_df = extract_world_bank_gdp()
    taxi_df = extract_nyc_taxi_sample(n_rows=5000)

    # SQLite round-trip
    titanic_from_db = round_trip_sqlite(titanic_df, "raw_titanic", db_path)

    # Validate all extractions
    datasets = {
        "titanic": titanic_df,
        "housing": housing_df,
        "gdp": gdp_df,
        "taxi": taxi_df,
        "titanic_sqlite": titanic_from_db,
    }
    for name, df in datasets.items():
        validate_extraction(df, name)

    # Save extracted data
    titanic_df.to_csv(output_dir / "titanic_raw.csv", index=False)
    housing_df.to_csv(output_dir / "housing_raw.csv", index=False)
    gdp_df.to_csv(output_dir / "gdp_raw.csv", index=False)
    taxi_df.to_parquet(output_dir / "taxi_raw.parquet", index=False)

    # Save metadata
    metadata = {
        "datasets": {
            name: {"rows": int(df.shape[0]), "columns": int(df.shape[1]), "columns_list": list(df.columns)}
            for name, df in datasets.items()
        }
    }
    with open(output_dir / "extraction_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[Done] All extracted data saved to {output_dir}")
    return datasets


if __name__ == "__main__":
    main()
