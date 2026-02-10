"""
Category 22: SageMaker Pipeline Runner
========================================
Orchestrate the full pipeline as a SageMaker Pipeline:
  data extraction -> Glue/S3/Athena -> engineering -> training -> inference

Prerequisites:
  - SageMaker domain, IAM roles, and S3 buckets already provisioned
  - Glue database, crawler role, and Athena workgroup already provisioned
  - SageMaker Python SDK installed (pip install sagemaker)

Usage:
    python 22_sagemaker_pipeline_runner.py

Note: This script is meant to run inside a SageMaker environment
(Studio notebook, Processing Job, or CI/CD pipeline).
"""

import json
from pathlib import Path


def build_pipeline_definition():
    """
    Build and return a SageMaker Pipeline definition.

    This function demonstrates the full pipeline construction.
    It requires the SageMaker SDK and an active AWS session to execute.
    """
    try:
        import sagemaker
        from sagemaker.processing import ProcessingInput, ProcessingOutput
        from sagemaker.sklearn import SKLearn, SKLearnProcessor
        from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger, ParameterString
        from sagemaker.workflow.pipeline import Pipeline
        from sagemaker.workflow.steps import ProcessingStep, TrainingStep, TransformStep
    except ImportError:
        print("[Pipeline] SageMaker SDK not installed. Printing pipeline definition only.")
        print_pipeline_definition()
        return None

    # --- Session setup ---
    session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    bucket = session.default_bucket()
    prefix = "ml-pipeline-practice"

    # --- Pipeline Parameters (configurable per run) ---
    instance_type = ParameterString(name="InstanceType", default_value="ml.m5.xlarge")
    n_estimators = ParameterInteger(name="NEstimators", default_value=200)
    max_depth = ParameterInteger(name="MaxDepth", default_value=5)
    learning_rate = ParameterFloat(name="LearningRate", default_value=0.1)

    # ===================================================================
    # Step 1: Data Extraction (Processing Job)
    # ===================================================================
    extract_processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type=instance_type,
        instance_count=1,
        base_job_name="step-extract",
        sagemaker_session=session,
    )

    extract_step = ProcessingStep(
        name="DataExtraction",
        processor=extract_processor,
        code="code_folder/22_data_extraction_pipeline.py",
        outputs=[
            ProcessingOutput(
                output_name="extracted_data",
                source="/opt/ml/processing/output",
                destination=f"s3://{bucket}/{prefix}/extracted/",
            )
        ],
    )

    # ===================================================================
    # Step 1B: Glue ETL / S3 / Athena (Processing Job)
    # ===================================================================
    glue_processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type=instance_type,
        instance_count=1,
        base_job_name="step-glue-athena",
        sagemaker_session=session,
    )

    glue_step = ProcessingStep(
        name="GlueS3Athena",
        processor=glue_processor,
        code="code_folder/22_glue_s3_athena_pipeline.py",
        inputs=[
            ProcessingInput(
                source=extract_step.properties.ProcessingOutputConfig.Outputs[
                    "extracted_data"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="glue_athena_output",
                source="/opt/ml/processing/output",
                destination=f"s3://{bucket}/{prefix}/glue-athena-output/",
            )
        ],
        job_arguments=[
            "--mode", "local",  # Use local mode inside Processing Job
        ],
    )
    glue_step.add_depends_on([extract_step])

    # ===================================================================
    # Step 2: Data Engineering (Processing Job)
    # ===================================================================
    engineer_processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type=instance_type,
        instance_count=1,
        base_job_name="step-engineer",
        sagemaker_session=session,
    )

    engineer_step = ProcessingStep(
        name="DataEngineering",
        processor=engineer_processor,
        code="code_folder/22_data_engineering_pipeline.py",
        inputs=[
            # Can use either raw extracted data or Glue/Athena cleaned output
            ProcessingInput(
                source=glue_step.properties.ProcessingOutputConfig.Outputs[
                    "glue_athena_output"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train_data",
                source="/opt/ml/processing/output",
                destination=f"s3://{bucket}/{prefix}/engineered/",
            )
        ],
    )
    engineer_step.add_depends_on([glue_step])

    # ===================================================================
    # Step 3: Model Training (Training Job)
    # ===================================================================
    estimator = SKLearn(
        entry_point="22_ml_model_training.py",
        source_dir="code_folder",
        role=role,
        instance_type=instance_type,
        instance_count=1,
        framework_version="1.2-1",
        hyperparameters={
            "n-estimators": n_estimators,
            "max-depth": max_depth,
            "learning-rate": learning_rate,
        },
        output_path=f"s3://{bucket}/{prefix}/model/",
        sagemaker_session=session,
    )

    train_step = TrainingStep(
        name="ModelTraining",
        estimator=estimator,
        inputs={
            "train": sagemaker.inputs.TrainingInput(
                s3_data=engineer_step.properties.ProcessingOutputConfig.Outputs[
                    "train_data"
                ].S3Output.S3Uri,
                content_type="text/csv",
            )
        },
    )
    train_step.add_depends_on([engineer_step])

    # ===================================================================
    # Step 4: Batch Inference (Processing Job with model)
    # ===================================================================
    inference_processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type=instance_type,
        instance_count=1,
        base_job_name="step-inference",
        sagemaker_session=session,
    )

    inference_step = ProcessingStep(
        name="InferenceOverlay",
        processor=inference_processor,
        code="code_folder/22_ml_inference_overlay.py",
        inputs=[
            ProcessingInput(
                source=engineer_step.properties.ProcessingOutputConfig.Outputs[
                    "train_data"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/input",
            ),
            ProcessingInput(
                source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/model",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="predictions",
                source="/opt/ml/processing/output",
                destination=f"s3://{bucket}/{prefix}/predictions/",
            )
        ],
    )
    inference_step.add_depends_on([train_step])

    # ===================================================================
    # Assemble Pipeline
    # ===================================================================
    pipeline = Pipeline(
        name="data-extraction-ml-pipeline",
        parameters=[instance_type, n_estimators, max_depth, learning_rate],
        steps=[extract_step, glue_step, engineer_step, train_step, inference_step],
        sagemaker_session=session,
    )

    print("[Pipeline] Pipeline definition built successfully")
    print(f"  Steps: DataExtraction -> GlueS3Athena -> DataEngineering -> ModelTraining -> InferenceOverlay")
    print(f"  S3 prefix: s3://{bucket}/{prefix}/")

    return pipeline


def print_pipeline_definition():
    """Print the pipeline structure for reference when SDK is not available."""
    definition = {
        "name": "data-extraction-ml-pipeline",
        "parameters": {
            "InstanceType": {"type": "String", "default": "ml.m5.xlarge"},
            "NEstimators": {"type": "Integer", "default": 200},
            "MaxDepth": {"type": "Integer", "default": 5},
            "LearningRate": {"type": "Float", "default": 0.1},
        },
        "steps": [
            {
                "name": "DataExtraction",
                "type": "Processing",
                "script": "code_folder/22_data_extraction_pipeline.py",
                "processor": "SKLearnProcessor (1.2-1)",
                "outputs": ["s3://<bucket>/ml-pipeline-practice/extracted/"],
            },
            {
                "name": "GlueS3Athena",
                "type": "Processing",
                "script": "code_folder/22_glue_s3_athena_pipeline.py",
                "processor": "SKLearnProcessor (1.2-1)",
                "inputs": ["DataExtraction.extracted_data"],
                "outputs": ["s3://<bucket>/ml-pipeline-practice/glue-athena-output/"],
                "depends_on": ["DataExtraction"],
                "description": "Upload to S3, simulate Glue ETL cleaning, Athena queries, downstream export",
            },
            {
                "name": "DataEngineering",
                "type": "Processing",
                "script": "code_folder/22_data_engineering_pipeline.py",
                "processor": "SKLearnProcessor (1.2-1)",
                "inputs": ["GlueS3Athena.glue_athena_output"],
                "outputs": ["s3://<bucket>/ml-pipeline-practice/engineered/"],
                "depends_on": ["GlueS3Athena"],
            },
            {
                "name": "ModelTraining",
                "type": "Training",
                "script": "code_folder/22_ml_model_training.py",
                "estimator": "SKLearn (1.2-1)",
                "inputs": ["DataEngineering.train_data"],
                "outputs": ["s3://<bucket>/ml-pipeline-practice/model/"],
                "depends_on": ["DataEngineering"],
            },
            {
                "name": "InferenceOverlay",
                "type": "Processing",
                "script": "code_folder/22_ml_inference_overlay.py",
                "processor": "SKLearnProcessor (1.2-1)",
                "inputs": [
                    "DataEngineering.train_data",
                    "ModelTraining.model_artifacts",
                ],
                "outputs": ["s3://<bucket>/ml-pipeline-practice/predictions/"],
                "depends_on": ["ModelTraining"],
            },
        ],
    }

    print("\n[Pipeline Definition]")
    print(json.dumps(definition, indent=2))


def execute_pipeline(pipeline):
    """Upsert and start the pipeline execution."""
    if pipeline is None:
        print("[Pipeline] Cannot execute — pipeline not built (SDK missing)")
        return None

    import sagemaker

    role = sagemaker.get_execution_role()

    # Upsert (create or update)
    pipeline.upsert(role_arn=role)
    print("[Pipeline] Pipeline upserted to SageMaker")

    # Start execution
    execution = pipeline.start()
    print(f"[Pipeline] Execution started: {execution.arn}")
    print("[Pipeline] Waiting for completion...")
    execution.wait()

    # Check status
    status = execution.describe()["PipelineExecutionStatus"]
    print(f"[Pipeline] Execution status: {status}")

    if status == "Succeeded":
        print("[Pipeline] Pipeline completed successfully!")
        steps = execution.list_steps()
        for step in steps:
            print(f"  -> {step['StepName']}: {step['StepStatus']}")
    else:
        print("[Pipeline] Pipeline did not succeed. Check CloudWatch logs.")

    return execution


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("SageMaker Pipeline: Data Extraction -> ML Inference")
    print("=" * 60)

    pipeline = build_pipeline_definition()

    if pipeline is not None:
        # Only execute if we're in a SageMaker environment
        import os

        if os.environ.get("AWS_DEFAULT_REGION") or os.path.isdir("/opt/ml"):
            execute_pipeline(pipeline)
        else:
            print("\n[Info] Not in SageMaker environment. Pipeline definition printed above.")
            print("[Info] To execute, run this script in a SageMaker Studio notebook or CI/CD pipeline.")
    else:
        print("\n[Info] Install sagemaker SDK to build and execute the pipeline:")
        print("  pip install sagemaker")


if __name__ == "__main__":
    main()
