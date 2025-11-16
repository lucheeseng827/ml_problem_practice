"""
AWS SageMaker Training and Deployment
======================================
Category 19: MLOps - Cloud-native ML training and deployment

Use cases: Scalable model training, managed endpoints, distributed training
Demonstrates: SageMaker training jobs, model deployment, endpoint inference
"""

import os
import boto3
import sagemaker
from sagemaker.sklearn import SKLearn, SKLearnModel
from sagemaker.session import Session
import pandas as pd
import numpy as np


def prepare_training_data():
    """Prepare data for SageMaker training"""
    print("\n1. Preparing Training Data")
    print("-" * 70)

    # Generate sample data
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)

    # Create DataFrame
    df = pd.DataFrame(X, columns=['f1', 'f2', 'f3', 'f4'])
    df['target'] = y

    print(f"Dataset shape: {df.shape}")
    print(f"Features: {df.columns.tolist()}")

    return df


def upload_data_to_s3(df, bucket_name, prefix="iris-dataset"):
    """Upload training data to S3"""
    print("\n2. Uploading Data to S3")
    print("-" * 70)

    try:
        # Initialize S3 client
        s3_client = boto3.client('s3')
        sagemaker_session = Session()

        # Save data locally
        train_file = '/tmp/train.csv'
        df.to_csv(train_file, index=False, header=False)

        # Upload to S3
        s3_input_path = f"s3://{bucket_name}/{prefix}/train.csv"
        sagemaker_session.upload_data(
            path=train_file,
            bucket=bucket_name,
            key_prefix=prefix
        )

        print(f"✓ Data uploaded to: {s3_input_path}")
        return s3_input_path

    except Exception as e:
        print(f"⚠️  S3 upload failed: {e}")
        print("Note: Set AWS credentials and create S3 bucket first")
        return None


def create_training_script():
    """Create training script for SageMaker"""
    training_script = '''
import argparse
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib


def main():
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=10)
    parser.add_argument('--min-samples-split', type=int, default=2)

    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))

    args, _ = parser.parse_known_args()

    # Load data
    train_data = pd.read_csv(os.path.join(args.train, 'train.csv'), header=None)
    X = train_data.iloc[:, :-1].values
    y = train_data.iloc[:, -1].values

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy:.4f}")

    # Save model
    joblib.dump(model, os.path.join(args.model_dir, 'model.joblib'))


if __name__ == '__main__':
    main()
'''
    return training_script


def run_sagemaker_training():
    """Run training job on SageMaker"""
    print("\n3. Creating SageMaker Training Job")
    print("-" * 70)

    try:
        # Initialize SageMaker session
        sagemaker_session = sagemaker.Session()
        role = sagemaker.get_execution_role()  # IAM role with SageMaker permissions

        # Define estimator
        sklearn_estimator = SKLearn(
            entry_point='train.py',  # Training script
            framework_version='1.0-1',
            instance_type='ml.m5.large',
            instance_count=1,
            role=role,
            sagemaker_session=sagemaker_session,
            hyperparameters={
                'n-estimators': 200,
                'max-depth': 15,
                'min-samples-split': 5
            },
            max_run=3600,  # Max training time in seconds
            output_path='s3://your-bucket/model-artifacts'
        )

        print("Training job configuration:")
        print(f"  • Instance type: ml.m5.large")
        print(f"  • Instance count: 1")
        print(f"  • Max runtime: 3600s")
        print(f"  • Framework: scikit-learn 1.0")

        # Start training
        print("\n4. Starting Training Job")
        print("-" * 70)
        # sklearn_estimator.fit({'train': 's3://your-bucket/iris-dataset/'})
        print("Command: estimator.fit({'train': 's3://bucket/data/'})")
        print("  • Job submitted to SageMaker")
        print("  • Training runs on dedicated compute")
        print("  • Logs streamed to CloudWatch")
        print("  • Model artifacts saved to S3")

        return sklearn_estimator

    except Exception as e:
        print(f"⚠️  SageMaker training setup failed: {e}")
        print("Note: Requires AWS credentials and SageMaker permissions")
        return None


def deploy_model_to_endpoint():
    """Deploy trained model to SageMaker endpoint"""
    print("\n5. Deploying Model to Endpoint")
    print("-" * 70)

    try:
        # Deploy model (after training completes)
        print("Command: predictor = estimator.deploy(")
        print("    initial_instance_count=1,")
        print("    instance_type='ml.m5.large',")
        print("    endpoint_name='iris-classifier-endpoint'")
        print(")")
        print()
        print("Deployment process:")
        print("  • Creates endpoint configuration")
        print("  • Provisions inference instance (ml.m5.large)")
        print("  • Loads model from S3")
        print("  • Starts endpoint (takes 5-10 minutes)")
        print("  • Returns predictor object for inference")

    except Exception as e:
        print(f"⚠️  Deployment setup failed: {e}")


def make_predictions():
    """Make predictions using SageMaker endpoint"""
    print("\n6. Making Predictions")
    print("-" * 70)

    # Sample input
    sample_data = np.array([[5.1, 3.5, 1.4, 0.2]])

    print(f"Sample input: {sample_data}")
    print()
    print("Code:")
    print(f"prediction = predictor.predict({sample_data.tolist()})")
    print()
    print("Expected output: [0]  # Iris Setosa")
    print()
    print("Endpoint features:")
    print("  • Auto-scaling based on traffic")
    print("  • Load balancing across instances")
    print("  • HTTPS encryption")
    print("  • CloudWatch metrics")
    print("  • Cost: ~$0.10/hour per instance")


def demonstrate_sagemaker_workflow():
    """Demonstrate complete SageMaker ML workflow"""
    print("=" * 70)
    print("AWS SageMaker Training and Deployment Workflow")
    print("=" * 70)

    # Show training script
    print("\nTraining Script (train.py):")
    print("-" * 70)
    script = create_training_script()
    print("  • Reads data from SM_CHANNEL_TRAIN")
    print("  • Trains RandomForest with hyperparameters")
    print("  • Saves model to SM_MODEL_DIR")
    print("  • SageMaker handles data/model I/O")

    # Prepare data
    df = prepare_training_data()

    # Show workflow steps
    run_sagemaker_training()
    deploy_model_to_endpoint()
    make_predictions()

    print("\n" + "=" * 70)
    print("SageMaker Advantages:")
    print("=" * 70)
    print("✓ Managed Infrastructure: No server management")
    print("✓ Scalability: Scale from 1 to 100+ instances")
    print("✓ Distributed Training: Multi-GPU, multi-node training")
    print("✓ Spot Instances: Up to 90% cost savings")
    print("✓ Built-in Algorithms: Pre-built algorithms for common tasks")
    print("✓ Model Registry: Version and track models")
    print("✓ Monitoring: Automatic model monitoring and drift detection")
    print("✓ AutoML: Automatic model selection and tuning (Autopilot)")
    print()

    print("Cost Optimization:")
    print("  • Use ml.t3.medium for development ($0.05/hour)")
    print("  • Use Spot instances for training (up to 90% savings)")
    print("  • Use inference autoscaling (scale to zero)")
    print("  • Use batch transform for offline inference")
    print("  • Delete endpoints when not in use")
    print()

    print("Production Best Practices:")
    print("  1. Store data in S3 with versioning")
    print("  2. Use script mode (not notebook mode)")
    print("  3. Parameterize hyperparameters")
    print("  4. Enable CloudWatch logging")
    print("  5. Use SageMaker Experiments for tracking")
    print("  6. Implement model validation before deployment")
    print("  7. Use A/B testing for new models")
    print("  8. Set up monitoring and alerts")
    print("  9. Use IAM roles with least privilege")
    print("  10. Tag resources for cost allocation")


def show_terraform_integration():
    """Show how to provision SageMaker with Terraform"""
    print("\n" + "=" * 70)
    print("Terraform Integration (see terraform/sagemaker/):")
    print("=" * 70)
    print("""
# SageMaker Notebook Instance
resource "aws_sagemaker_notebook_instance" "ml_notebook" {
  name          = "ml-practice-notebook"
  role_arn      = aws_iam_role.sagemaker_role.arn
  instance_type = "ml.t3.medium"
}

# SageMaker Model
resource "aws_sagemaker_model" "iris_model" {
  name               = "iris-classifier"
  execution_role_arn = aws_iam_role.sagemaker_role.arn

  primary_container {
    image = "your-ecr-repo/sklearn:latest"
    model_data_url = "s3://your-bucket/model.tar.gz"
  }
}

# SageMaker Endpoint Configuration
resource "aws_sagemaker_endpoint_configuration" "config" {
  name = "iris-classifier-config"

  production_variants {
    variant_name           = "primary"
    model_name             = aws_sagemaker_model.iris_model.name
    instance_type          = "ml.m5.large"
    initial_instance_count = 1
  }
}

# SageMaker Endpoint
resource "aws_sagemaker_endpoint" "endpoint" {
  name                 = "iris-classifier-endpoint"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.config.name
}
""")


def main():
    demonstrate_sagemaker_workflow()
    show_terraform_integration()

    print("\n" + "=" * 70)
    print("Setup Requirements:")
    print("=" * 70)
    print("1. Install AWS CLI and configure credentials:")
    print("   aws configure")
    print()
    print("2. Install SageMaker Python SDK:")
    print("   pip install sagemaker boto3")
    print()
    print("3. Create S3 bucket for data/models:")
    print("   aws s3 mb s3://your-ml-bucket")
    print()
    print("4. Create IAM role with SageMaker permissions")
    print()
    print("5. Run Terraform to provision infrastructure:")
    print("   cd terraform/sagemaker && terraform apply")


if __name__ == "__main__":
    main()
