"""
Apache Airflow ML Pipeline - Complete Model Lifecycle
======================================================
Category 19: MLOps - Orchestrated ML pipeline with scheduling

Use cases: Automated ML workflows, scheduled retraining, production pipelines
Demonstrates: Airflow DAGs, task dependencies, pipeline orchestration, scheduling
"""

from datetime import datetime, timedelta
from typing import Dict, Any
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib


# =========================================================================
# This is a standalone script that demonstrates Airflow DAG concepts
# In production, this would be split into actual Airflow DAG files
# =========================================================================


class MLPipelineSimulator:
    """Simulates an Airflow ML pipeline"""

    def __init__(self, pipeline_config: Dict[str, Any]):
        self.config = pipeline_config
        self.data_path = pipeline_config.get('data_path', '/tmp/data.csv')
        self.model_path = pipeline_config.get('model_path', '/tmp/model.pkl')
        self.metadata_path = pipeline_config.get('metadata_path', '/tmp/metadata.json')
        self.metrics = {}

    def task_data_ingestion(self, **context):
        """
        Task 1: Data Ingestion
        Simulates loading data from various sources
        """
        print("=" * 70)
        print("TASK 1: Data Ingestion")
        print("=" * 70)

        # Simulate data ingestion from database/API/S3
        from sklearn.datasets import load_iris
        X, y = load_iris(return_X_y=True)

        df = pd.DataFrame(X, columns=['f1', 'f2', 'f3', 'f4'])
        df['target'] = y

        # Save to temporary location
        df.to_csv(self.data_path, index=False)

        print(f"✓ Ingested {len(df)} rows")
        print(f"✓ Saved to: {self.data_path}")

        # Push metadata to XCom (Airflow's inter-task communication)
        metadata = {
            'row_count': len(df),
            'columns': list(df.columns),
            'timestamp': datetime.now().isoformat()
        }

        return metadata

    def task_data_validation(self, **context):
        """
        Task 2: Data Validation
        Validates data quality using Great Expectations concepts
        """
        print("\n" + "=" * 70)
        print("TASK 2: Data Validation")
        print("=" * 70)

        # Load data
        df = pd.read_csv(self.data_path)

        # Validation checks
        validations = {
            'no_nulls': df.isnull().sum().sum() == 0,
            'correct_shape': len(df) > 100,
            'target_values_valid': df['target'].isin([0, 1, 2]).all(),
            'feature_ranges_valid': (df[['f1', 'f2', 'f3', 'f4']] >= 0).all().all()
        }

        all_passed = all(validations.values())

        for check, passed in validations.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status}: {check}")

        if not all_passed:
            raise ValueError("Data validation failed!")

        print("\n✓ All validations passed")
        return validations

    def task_data_preprocessing(self, **context):
        """
        Task 3: Data Preprocessing
        Prepares data for training
        """
        print("\n" + "=" * 70)
        print("TASK 3: Data Preprocessing")
        print("=" * 70)

        df = pd.read_csv(self.data_path)

        # Split features and target
        X = df.drop('target', axis=1)
        y = df['target']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"✓ Train set: {len(X_train)} samples")
        print(f"✓ Test set: {len(X_test)} samples")

        # Save preprocessed data
        np.save('/tmp/X_train.npy', X_train)
        np.save('/tmp/X_test.npy', X_test)
        np.save('/tmp/y_train.npy', y_train)
        np.save('/tmp/y_test.npy', y_test)

        return {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'features': list(X.columns)
        }

    def task_model_training(self, **context):
        """
        Task 4: Model Training
        Trains ML model with hyperparameters
        """
        print("\n" + "=" * 70)
        print("TASK 4: Model Training")
        print("=" * 70)

        # Load preprocessed data
        X_train = np.load('/tmp/X_train.npy')
        y_train = np.load('/tmp/y_train.npy')

        # Get hyperparameters from config
        hyperparams = self.config.get('hyperparameters', {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5
        })

        print(f"Hyperparameters: {hyperparams}")

        # Train model
        model = RandomForestClassifier(**hyperparams, random_state=42)
        model.fit(X_train, y_train)

        # Save model
        joblib.dump(model, self.model_path)

        print(f"✓ Model trained")
        print(f"✓ Model saved to: {self.model_path}")

        return {'model_path': self.model_path}

    def task_model_evaluation(self, **context):
        """
        Task 5: Model Evaluation
        Evaluates model performance
        """
        print("\n" + "=" * 70)
        print("TASK 5: Model Evaluation")
        print("=" * 70)

        # Load model and test data
        model = joblib.load(self.model_path)
        X_test = np.load('/tmp/X_test.npy')
        y_test = np.load('/tmp/y_test.npy')

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'f1_score': float(f1_score(y_test, y_pred, average='weighted')),
            'timestamp': datetime.now().isoformat()
        }

        self.metrics = metrics

        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")

        # Save metrics
        with open('/tmp/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        return metrics

    def task_model_validation_gate(self, **context):
        """
        Task 6: Model Validation Gate
        Decides if model is good enough for deployment
        """
        print("\n" + "=" * 70)
        print("TASK 6: Model Validation Gate")
        print("=" * 70)

        # Load metrics
        with open('/tmp/metrics.json', 'r') as f:
            metrics = json.load(f)

        # Define thresholds
        min_accuracy = self.config.get('min_accuracy', 0.85)
        min_f1_score = self.config.get('min_f1_score', 0.85)

        # Validation checks
        accuracy_pass = metrics['accuracy'] >= min_accuracy
        f1_pass = metrics['f1_score'] >= min_f1_score

        print(f"Accuracy: {metrics['accuracy']:.4f} (threshold: {min_accuracy})")
        print(f"  {'✓ PASS' if accuracy_pass else '✗ FAIL'}")
        print(f"F1 Score: {metrics['f1_score']:.4f} (threshold: {min_f1_score})")
        print(f"  {'✓ PASS' if f1_pass else '✗ FAIL'}")

        if not (accuracy_pass and f1_pass):
            print("\n✗ Model did not meet quality gates!")
            print("  Pipeline will SKIP deployment")
            return {'deploy': False, 'reason': 'quality_gate_failed'}

        print("\n✓ Model passed quality gates")
        print("  Pipeline will PROCEED to deployment")
        return {'deploy': True}

    def task_model_registration(self, **context):
        """
        Task 7: Model Registration
        Registers model in MLflow/SageMaker model registry
        """
        print("\n" + "=" * 70)
        print("TASK 7: Model Registration")
        print("=" * 70)

        # Check if we should proceed
        validation_result = context.get('ti').xcom_pull(task_ids='model_validation_gate')
        if not validation_result.get('deploy', False):
            print("✗ Skipping registration - quality gate failed")
            return {'registered': False}

        # Simulate MLflow registration
        model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        print(f"Registering model version: {model_version}")
        print("  • Adding to model registry")
        print("  • Tagging with experiment metadata")
        print("  • Transitioning to 'Staging'")

        metadata = {
            'model_version': model_version,
            'registered_at': datetime.now().isoformat(),
            'stage': 'Staging',
            'metrics': self.metrics
        }

        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Model registered: {model_version}")
        return metadata

    def task_model_deployment(self, **context):
        """
        Task 8: Model Deployment
        Deploys model to production endpoint
        """
        print("\n" + "=" * 70)
        print("TASK 8: Model Deployment")
        print("=" * 70)

        # Check if we should proceed
        validation_result = context.get('ti').xcom_pull(task_ids='model_validation_gate')
        if not validation_result.get('deploy', False):
            print("✗ Skipping deployment - quality gate failed")
            return {'deployed': False}

        # Load model metadata
        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)

        model_version = metadata['model_version']

        print(f"Deploying model: {model_version}")
        print("  • Creating deployment configuration")
        print("  • Updating endpoint with new model")
        print("  • Running health checks")
        print("  • Transitioning registry stage: Staging → Production")

        # Simulate deployment
        deployment_info = {
            'endpoint_url': 'https://api.example.com/predict',
            'model_version': model_version,
            'deployed_at': datetime.now().isoformat(),
            'status': 'active'
        }

        print(f"✓ Model deployed to: {deployment_info['endpoint_url']}")
        print(f"✓ Version: {model_version}")
        return deployment_info

    def task_send_notification(self, **context):
        """
        Task 9: Send Notification
        Notifies team of pipeline completion
        """
        print("\n" + "=" * 70)
        print("TASK 9: Send Notification")
        print("=" * 70)

        # Gather pipeline results
        validation_result = context.get('ti').xcom_pull(task_ids='model_validation_gate')

        if validation_result.get('deploy', False):
            metadata = context.get('ti').xcom_pull(task_ids='model_registration')
            print("✉️  Sending SUCCESS notification:")
            print(f"  • Model version: {metadata['model_version']}")
            print(f"  • Metrics: Accuracy={self.metrics['accuracy']:.4f}")
            print(f"  • Status: DEPLOYED")
        else:
            print("✉️  Sending FAILURE notification:")
            print(f"  • Reason: Quality gate failed")
            print(f"  • Metrics: Accuracy={self.metrics['accuracy']:.4f}")
            print(f"  • Status: NOT DEPLOYED")

        print("✓ Notification sent")
        return {'notified': True}


def create_airflow_dag_definition():
    """
    Creates the actual Airflow DAG definition
    This is what you would put in your dags/ folder
    """
    dag_definition = '''
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime, timedelta

# Import your ML pipeline functions
from ml_pipeline_tasks import (
    data_ingestion,
    data_validation,
    data_preprocessing,
    model_training,
    model_evaluation,
    model_validation_gate,
    model_registration,
    model_deployment,
    send_notification
)

# Default arguments
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'email': ['ml-team@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
}

# Define DAG
dag = DAG(
    'ml_model_training_pipeline',
    default_args=default_args,
    description='Complete ML model lifecycle pipeline',
    schedule_interval='0 2 * * 1',  # Weekly on Monday at 2 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'production', 'weekly'],
)

# Define tasks
task_1 = PythonOperator(
    task_id='data_ingestion',
    python_callable=data_ingestion,
    dag=dag,
)

task_2 = PythonOperator(
    task_id='data_validation',
    python_callable=data_validation,
    dag=dag,
)

task_3 = PythonOperator(
    task_id='data_preprocessing',
    python_callable=data_preprocessing,
    dag=dag,
)

task_4 = PythonOperator(
    task_id='model_training',
    python_callable=model_training,
    dag=dag,
)

task_5 = PythonOperator(
    task_id='model_evaluation',
    python_callable=model_evaluation,
    dag=dag,
)

task_6 = BranchPythonOperator(
    task_id='model_validation_gate',
    python_callable=model_validation_gate,
    dag=dag,
)

task_7 = PythonOperator(
    task_id='model_registration',
    python_callable=model_registration,
    dag=dag,
)

task_8 = PythonOperator(
    task_id='model_deployment',
    python_callable=model_deployment,
    dag=dag,
)

task_9 = PythonOperator(
    task_id='send_notification',
    python_callable=send_notification,
    trigger_rule='all_done',  # Run even if upstream fails
    dag=dag,
)

# Define task dependencies
task_1 >> task_2 >> task_3 >> task_4 >> task_5 >> task_6
task_6 >> task_7 >> task_8 >> task_9
'''
    return dag_definition


def main():
    """
    Simulates running the complete ML pipeline
    """
    print("=" * 70)
    print("APACHE AIRFLOW ML PIPELINE - COMPLETE MODEL LIFECYCLE")
    print("=" * 70)
    print()
    print("This simulates an Airflow DAG execution")
    print("In production, this runs on Airflow scheduler")
    print()

    # Pipeline configuration
    config = {
        'data_path': '/tmp/ml_pipeline_data.csv',
        'model_path': '/tmp/ml_pipeline_model.pkl',
        'metadata_path': '/tmp/ml_pipeline_metadata.json',
        'hyperparameters': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5
        },
        'min_accuracy': 0.85,
        'min_f1_score': 0.85
    }

    # Create pipeline simulator
    pipeline = MLPipelineSimulator(config)

    # Simulate task context (Airflow provides this)
    context = {'ti': type('obj', (object,), {
        'xcom_pull': lambda task_ids: pipeline.metrics if 'evaluation' in task_ids else {'deploy': True}
    })()}

    # Execute pipeline tasks in order
    try:
        # Task 1-5: Data pipeline and training
        pipeline.task_data_ingestion(**context)
        pipeline.task_data_validation(**context)
        pipeline.task_data_preprocessing(**context)
        pipeline.task_model_training(**context)
        pipeline.task_model_evaluation(**context)

        # Task 6: Quality gate
        gate_result = pipeline.task_model_validation_gate(**context)
        context['ti'].xcom_pull = lambda task_ids: gate_result if 'gate' in task_ids else pipeline.metrics

        # Task 7-9: Deployment pipeline
        pipeline.task_model_registration(**context)
        pipeline.task_model_deployment(**context)
        pipeline.task_send_notification(**context)

        print("\n" + "=" * 70)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 70)

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"✗ PIPELINE FAILED: {e}")
        print("=" * 70)
        raise

    # Show DAG definition
    print("\n" + "=" * 70)
    print("AIRFLOW DAG DEFINITION")
    print("=" * 70)
    print("\nSave this to: dags/ml_model_training_pipeline.py")
    print()
    print(create_airflow_dag_definition())

    print("\n" + "=" * 70)
    print("KEY CONCEPTS")
    print("=" * 70)
    print("""
1. **DAG (Directed Acyclic Graph)**: Defines task dependencies
2. **Tasks**: Individual units of work (Python functions)
3. **Operators**: Define how tasks execute (PythonOperator, BashOperator, etc.)
4. **XCom**: Inter-task communication for passing data
5. **Schedule**: Cron-like scheduling (daily, weekly, etc.)
6. **Retries**: Automatic retry on failure
7. **Branching**: Conditional task execution based on results
8. **Triggers**: Control when tasks run (all_done, all_success, etc.)

SCHEDULING OPTIONS:
  • '@daily': Run once per day
  • '@weekly': Run once per week
  • '0 2 * * *': Cron expression (2 AM daily)
  • '@hourly': Run every hour
  • None: Manual trigger only

DEPLOYMENT:
  1. Save DAG to airflow/dags/ folder
  2. Airflow scheduler picks it up automatically
  3. View in Airflow UI: http://localhost:8080
  4. Trigger manually or wait for schedule
  5. Monitor execution in UI
  6. View logs for each task
    """)


if __name__ == "__main__":
    main()
