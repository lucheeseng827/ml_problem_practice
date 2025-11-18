"""
Great Expectations Data Validation
===================================
Category 19: MLOps - Data quality and validation

Use cases: Data validation, schema enforcement, data profiling, pipeline testing
Demonstrates: Expectations, validation, data docs, checkpoints
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def create_sample_data(with_issues=False):
    """Create sample ML training data"""
    np.random.seed(42)
    n_samples = 1000

    data = {
        'user_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.lognormal(10.5, 0.5, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'loan_amount': np.random.uniform(1000, 50000, n_samples),
        'employment_length': np.random.randint(0, 40, n_samples),
        'timestamp': [datetime.now() - timedelta(days=np.random.randint(0, 365))
                     for _ in range(n_samples)],
        'default': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    }

    df = pd.DataFrame(data)

    if with_issues:
        # Introduce data quality issues
        df.loc[10:15, 'age'] = -1  # Invalid age
        df.loc[20:25, 'credit_score'] = 1000  # Invalid credit score
        df.loc[30:32, 'income'] = np.nan  # Missing values
        df.loc[40, 'user_id'] = df.loc[41, 'user_id']  # Duplicate ID

    return df


def demonstrate_expectations():
    """Demonstrate Great Expectations validation"""
    print("=" * 70)
    print("Great Expectations - Data Quality & Validation")
    print("=" * 70)

    print("\n1. Creating Sample Dataset")
    print("-" * 70)
    df = create_sample_data(with_issues=False)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())

    print("\n2. Defining Data Expectations")
    print("-" * 70)
    expectations = {
        "user_id": [
            "expect_column_values_to_be_unique",
            "expect_column_values_to_not_be_null"
        ],
        "age": [
            "expect_column_values_to_be_between(18, 100)",
            "expect_column_values_to_not_be_null"
        ],
        "income": [
            "expect_column_values_to_be_between(0, 1000000)",
            "expect_column_values_to_not_be_null"
        ],
        "credit_score": [
            "expect_column_values_to_be_between(300, 850)",
            "expect_column_min_to_be_between(300, 400)",
            "expect_column_max_to_be_between(800, 850)"
        ],
        "loan_amount": [
            "expect_column_values_to_be_between(0, 100000)"
        ],
        "default": [
            "expect_column_values_to_be_in_set([0, 1])"
        ]
    }

    print("Defined expectations:")
    for col, exps in expectations.items():
        print(f"\n  {col}:")
        for exp in exps:
            print(f"    - {exp}")

    print("\n3. Manual Validation (without Great Expectations)")
    print("-" * 70)

    validation_results = {}

    # User ID uniqueness
    is_unique = df['user_id'].is_unique
    validation_results['user_id_unique'] = is_unique
    print(f"✓ user_id unique: {is_unique}")

    # Age range
    age_valid = df['age'].between(18, 100).all()
    validation_results['age_valid'] = age_valid
    print(f"✓ age in [18, 100]: {age_valid}")

    # Credit score range
    credit_valid = df['credit_score'].between(300, 850).all()
    validation_results['credit_valid'] = credit_valid
    print(f"✓ credit_score in [300, 850]: {credit_valid}")

    # Null checks
    null_counts = df.isnull().sum()
    validation_results['null_counts'] = null_counts.to_dict()
    print(f"✓ Null values per column:")
    for col, count in null_counts.items():
        if count > 0:
            print(f"    {col}: {count}")

    all_valid = all([
        is_unique,
        age_valid,
        credit_valid,
        null_counts.sum() == 0
    ])
    print(f"\n{'✓' if all_valid else '✗'} Overall validation: {'PASSED' if all_valid else 'FAILED'}")

    print("\n4. Testing with Invalid Data")
    print("-" * 70)
    df_with_issues = create_sample_data(with_issues=True)

    print("Introduced data quality issues:")
    print("  • Invalid ages (-1)")
    print("  • Invalid credit scores (> 850)")
    print("  • Missing income values")
    print("  • Duplicate user IDs")

    # Validate
    issues_found = []

    if not df_with_issues['user_id'].is_unique:
        duplicates = df_with_issues[df_with_issues.duplicated(['user_id'], keep=False)]
        issues_found.append(f"Duplicate user_ids: {len(duplicates)}")

    invalid_ages = df_with_issues[~df_with_issues['age'].between(18, 100)]
    if len(invalid_ages) > 0:
        issues_found.append(f"Invalid ages: {len(invalid_ages)} rows")

    invalid_credit = df_with_issues[~df_with_issues['credit_score'].between(300, 850)]
    if len(invalid_credit) > 0:
        issues_found.append(f"Invalid credit scores: {len(invalid_credit)} rows")

    null_counts = df_with_issues.isnull().sum()
    if null_counts.sum() > 0:
        issues_found.append(f"Missing values: {null_counts.sum()} total")

    print(f"\n✗ Validation FAILED with {len(issues_found)} issue types:")
    for issue in issues_found:
        print(f"  • {issue}")

    print("\n5. Great Expectations Workflow")
    print("-" * 70)
    print("""
# Initialize Great Expectations
import great_expectations as ge

# Create context
context = ge.get_context()

# Add datasource
datasource = context.sources.add_pandas("my_datasource")
data_asset = datasource.add_dataframe_asset(name="train_data")

# Create expectation suite
suite = context.add_expectation_suite("loan_data_validation")

# Add expectations
validator = context.get_validator(
    batch_request=data_asset.build_batch_request(dataframe=df),
    expectation_suite_name="loan_data_validation"
)

# Define expectations
validator.expect_column_values_to_be_unique("user_id")
validator.expect_column_values_to_be_between("age", 18, 100)
validator.expect_column_values_to_be_between("credit_score", 300, 850)
validator.expect_column_values_to_not_be_null("income")

# Save suite
validator.save_expectation_suite(discard_failed_expectations=False)

# Run validation
results = validator.validate()

# Generate data docs
context.build_data_docs()

# Results
print(f"Validation success: {results['success']}")
print(f"Expectations evaluated: {results['statistics']['evaluated_expectations']}")
print(f"Successful expectations: {results['statistics']['successful_expectations']}")
""")

    print("\n6. Checkpoint for CI/CD Integration")
    print("-" * 70)
    print("""
# Create checkpoint
checkpoint = context.add_checkpoint(
    name="daily_data_validation",
    validations=[{
        "batch_request": data_asset.build_batch_request(),
        "expectation_suite_name": "loan_data_validation"
    }]
)

# Run checkpoint
results = checkpoint.run()

# In CI/CD pipeline (GitHub Actions, Jenkins, etc.)
if not results["success"]:
    raise ValueError("Data validation failed!")
    # This fails the pipeline and prevents bad data from training
""")

    print("\n" + "=" * 70)
    print("Key Great Expectations Features:")
    print("=" * 70)
    print("✓ Expectations Library: 50+ built-in expectations")
    print("✓ Auto-Profiling: Automatically generate expectations from data")
    print("✓ Data Docs: Beautiful HTML documentation of validation results")
    print("✓ Checkpoints: Reusable validation workflows")
    print("✓ Multiple Backends: Pandas, Spark, SQL databases")
    print("✓ Integrations: Airflow, Prefect, Dagster, dbt")
    print("✓ Custom Expectations: Define domain-specific validations")
    print()

    print("Common Expectations:")
    print("  • expect_column_values_to_be_unique()")
    print("  • expect_column_values_to_not_be_null()")
    print("  • expect_column_values_to_be_between(min, max)")
    print("  • expect_column_values_to_be_in_set([values])")
    print("  • expect_column_values_to_match_regex(pattern)")
    print("  • expect_table_row_count_to_be_between(min, max)")
    print("  • expect_column_mean_to_be_between(min, max)")
    print("  • expect_column_distinct_values_to_be_in_set()")
    print()

    print("Production Workflow:")
    print("  1. Define expectations during exploratory data analysis")
    print("  2. Create expectation suite for each data source")
    print("  3. Run validation in data pipeline")
    print("  4. Generate data docs for team visibility")
    print("  5. Integrate checkpoints into CI/CD")
    print("  6. Alert on validation failures")
    print("  7. Track data quality metrics over time")
    print()

    print("Benefits:")
    print("  • Catch data issues before training")
    print("  • Document data assumptions")
    print("  • Enable data testing in CI/CD")
    print("  • Improve model reliability")
    print("  • Facilitate collaboration between teams")


def show_integration_with_ml_pipeline():
    """Show how to integrate with ML pipeline"""
    print("\n" + "=" * 70)
    print("Integration with ML Pipeline:")
    print("=" * 70)
    print("""
# In your training pipeline
def train_model(data_path):
    # Load data
    df = pd.read_csv(data_path)

    # Validate data
    results = run_validation_checkpoint(df, "train_data_validation")

    if not results["success"]:
        send_alert("Data validation failed for training data")
        raise ValueError("Training aborted due to data quality issues")

    # Proceed with training
    model = train_model_logic(df)

    return model


# In your inference pipeline
def predict(input_data):
    # Validate input
    results = run_validation_checkpoint(input_data, "inference_data_validation")

    if not results["success"]:
        log_warning("Input data quality issues detected")
        # Handle gracefully (return error, use defaults, etc.)

    predictions = model.predict(input_data)
    return predictions
""")


def main():
    demonstrate_expectations()
    show_integration_with_ml_pipeline()

    print("\n" + "=" * 70)
    print("Setup Instructions:")
    print("=" * 70)
    print("# Install Great Expectations")
    print("pip install great-expectations")
    print()
    print("# Initialize project")
    print("great_expectations init")
    print()
    print("# This creates:")
    print("great_expectations/")
    print("├── great_expectations.yml")
    print("├── expectations/")
    print("├── checkpoints/")
    print("├── plugins/")
    print("└── uncommitted/")
    print()
    print("# View data docs")
    print("great_expectations docs build")


if __name__ == "__main__":
    main()
