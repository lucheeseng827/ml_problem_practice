"""
Category 22: Data Engineering Pipeline
=======================================
Clean, transform, and feature-engineer extracted data.
Uses Titanic dataset as the primary example.

Stages:
  1. Load extracted data
  2. Handle missing values
  3. Type casting and cleanup
  4. Feature engineering
  5. Encoding categorical variables
  6. Normalization / scaling
  7. Train / test split with stratification
  8. Save engineered data

Usage:
    python 22_data_engineering_pipeline.py

SageMaker Processing Job:
    Input:  /opt/ml/processing/input/
    Output: /opt/ml/processing/output/
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def get_input_dir() -> Path:
    """Return input directory — uses SageMaker path if available."""
    if os.path.isdir("/opt/ml/processing/input"):
        return Path("/opt/ml/processing/input")
    return Path(__file__).parent.parent / "data" / "extracted"


def get_output_dir() -> Path:
    """Return output directory — uses SageMaker path if available."""
    sm_output = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/processing/output")
    if os.path.isdir("/opt/ml"):
        output_dir = Path(sm_output)
    else:
        output_dir = Path(__file__).parent.parent / "data" / "engineered"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ---------------------------------------------------------------------------
# 1. Load Extracted Data
# ---------------------------------------------------------------------------
def load_extracted_data(input_dir: Path) -> pd.DataFrame:
    """Load raw Titanic CSV from extraction stage."""
    csv_path = input_dir / "titanic_raw.csv"
    print(f"[Load] Reading {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  -> {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  -> Columns: {list(df.columns)}")
    print(f"  -> Null counts:\n{df.isnull().sum().to_string()}")
    return df


# ---------------------------------------------------------------------------
# 2. Handle Missing Values
# ---------------------------------------------------------------------------
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values: median for numeric, mode for categorical."""
    print("[Engineer] Handling missing values")
    df = df.copy()

    # Numeric columns: fill with median
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  -> {col}: filled {null_count} nulls with median={median_val:.2f}")

    # Categorical columns: fill with mode
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"  -> {col}: filled {null_count} nulls with mode='{mode_val}'")

    remaining_nulls = df.isnull().sum().sum()
    print(f"  -> Remaining nulls: {remaining_nulls}")
    return df


# ---------------------------------------------------------------------------
# 3. Type Casting and Cleanup
# ---------------------------------------------------------------------------
def type_cast_and_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure correct types and clean string columns."""
    print("[Engineer] Type casting and cleanup")
    df = df.copy()

    # Ensure numeric types
    if "Age" in df.columns:
        df["Age"] = df["Age"].astype(float)
    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].astype(float)

    # Strip whitespace from string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
        print(f"  -> Stripped whitespace from '{col}'")

    print(f"  -> Dtypes:\n{df.dtypes.to_string()}")
    return df


# ---------------------------------------------------------------------------
# 4. Feature Engineering
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features from existing columns."""
    print("[Engineer] Creating new features")
    df = df.copy()

    # Age binning
    if "Age" in df.columns:
        df["Age_Group"] = pd.cut(
            df["Age"],
            bins=[0, 12, 18, 35, 60, 100],
            labels=["Child", "Teen", "Adult", "Middle", "Senior"],
        )
        print("  -> Created Age_Group (binned)")

    # Family size (Titanic-specific)
    sib_col = "Siblings/Spouses Aboard"
    par_col = "Parents/Children Aboard"
    if sib_col in df.columns and par_col in df.columns:
        df["Family_Size"] = df[sib_col] + df[par_col] + 1
        df["Is_Alone"] = (df["Family_Size"] == 1).astype(int)
        print("  -> Created Family_Size, Is_Alone")

    # Fare per person
    if "Fare" in df.columns and sib_col in df.columns:
        df["Fare_Per_Person"] = df["Fare"] / (df[sib_col] + 1)
        print("  -> Created Fare_Per_Person")

    # Log transform for skewed features
    if "Fare" in df.columns:
        df["Log_Fare"] = np.log1p(df["Fare"])
        print("  -> Created Log_Fare (log-transformed)")

    # Name-based title extraction
    if "Name" in df.columns:
        df["Title"] = df["Name"].str.extract(r",\s*([^.]+)\.", expand=False)
        # Group rare titles
        common_titles = ["Mr", "Mrs", "Miss", "Master"]
        df["Title"] = df["Title"].apply(lambda x: x if x in common_titles else "Other")
        print(f"  -> Extracted Title from Name: {df['Title'].value_counts().to_dict()}")

    print(f"  -> Total columns after feature engineering: {df.shape[1]}")
    return df


# ---------------------------------------------------------------------------
# 5. Encode Categorical Variables
# ---------------------------------------------------------------------------
def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Encode categorical variables; return df and encoder mappings."""
    print("[Engineer] Encoding categorical variables")
    df = df.copy()
    encoders = {}

    # Label encode binary columns
    if "Sex" in df.columns:
        le = LabelEncoder()
        df["Sex_Encoded"] = le.fit_transform(df["Sex"])
        encoders["Sex"] = dict(zip(le.classes_, le.transform(le.classes_)))
        print(f"  -> Label encoded Sex: {encoders['Sex']}")

    # One-hot encode multi-class columns
    ohe_columns = []
    if "Age_Group" in df.columns:
        ohe_columns.append("Age_Group")
    if "Title" in df.columns:
        ohe_columns.append("Title")
    if "Pclass" in df.columns:
        df["Pclass"] = df["Pclass"].astype(str)
        ohe_columns.append("Pclass")

    if ohe_columns:
        df = pd.get_dummies(df, columns=ohe_columns, drop_first=True, dtype=int)
        print(f"  -> One-hot encoded: {ohe_columns}")

    # Drop original text columns no longer needed
    drop_cols = [c for c in ["Name", "Sex"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"  -> Dropped original text columns: {drop_cols}")

    print(f"  -> Final columns ({df.shape[1]}): {list(df.columns)}")
    return df, encoders


# ---------------------------------------------------------------------------
# 6. Normalize / Scale
# ---------------------------------------------------------------------------
def scale_features(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, StandardScaler]:
    """Standard-scale numeric features (excluding target)."""
    print("[Engineer] Scaling numeric features")
    df = df.copy()

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print(f"  -> Scaled {len(numeric_cols)} numeric columns with StandardScaler")
    return df, scaler


# ---------------------------------------------------------------------------
# 7. Train / Test Split
# ---------------------------------------------------------------------------
def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split into train/test with stratification."""
    print(f"[Engineer] Splitting data (test_size={test_size})")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    print(f"  -> Train: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows")
    print(f"  -> Target distribution (train): {y_train.value_counts(normalize=True).to_dict()}")
    print(f"  -> Target distribution (test):  {y_test.value_counts(normalize=True).to_dict()}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    input_dir = get_input_dir()
    output_dir = get_output_dir()
    target_col = "Survived"

    # Pipeline stages
    df = load_extracted_data(input_dir)
    df = handle_missing_values(df)
    df = type_cast_and_cleanup(df)
    df = engineer_features(df)
    df, encoders = encode_categoricals(df)
    df, scaler = scale_features(df, target_col)
    X_train, X_test, y_train, y_test = split_data(df, target_col)

    # Save engineered data
    train_df = X_train.copy()
    train_df[target_col] = y_train
    test_df = X_test.copy()
    test_df[target_col] = y_test

    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    # Save feature list and metadata
    metadata = {
        "target": target_col,
        "features": list(X_train.columns),
        "n_features": X_train.shape[1],
        "train_rows": X_train.shape[0],
        "test_rows": X_test.shape[0],
        "encoders": encoders,
    }
    with open(output_dir / "engineering_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[Done] Engineered data saved to {output_dir}")
    print(f"  -> train.csv: {train_df.shape}")
    print(f"  -> test.csv:  {test_df.shape}")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    main()
