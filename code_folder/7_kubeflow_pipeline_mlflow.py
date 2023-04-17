import kfp
from kfp import dsl
from kfp.components import load_component_from_file

# Load MLflow components
mlflow_log_component = load_component_from_file("mlflow_log_component.yaml")
mlflow_train_component = load_component_from_file("mlflow_train_component.yaml")
mlflow_test_component = load_component_from_file("mlflow_test_component.yaml")
mlflow_validate_component = load_component_from_file("mlflow_validate_component.yaml")


# Define pipeline
@dsl.pipeline(
    name="MLflow Training and Validation",
    description="A pipeline that logs an MLflow experiment, starts training, and performs model testing and validation.",
)
def mlflow_pipeline(
    experiment_name: str = "my_experiment",
    data_path: str = "gs://my-bucket/data",
    model_path: str = "gs://my-bucket/models",
    epochs: int = 10,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    validation_data_path: str = "gs://my-bucket/validation",
    test_data_path: str = "gs://my-bucket/test",
):
    # Log experiment
    mlflow_log = mlflow_log_component(experiment_name=experiment_name)

    # Train model
    mlflow_train = mlflow_train_component(
        data_path=data_path,
        model_path=model_path,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
    ).after(mlflow_log)

    # Test model
    mlflow_test = mlflow_test_component(
        data_path=test_data_path, model_path=model_path
    ).after(mlflow_train)

    # Validate model
    mlflow_validate = mlflow_validate_component(
        data_path=validation_data_path, model_path=model_path
    ).after(mlflow_test)


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(mlflow_pipeline, "mlflow_pipeline.tar.gz")
