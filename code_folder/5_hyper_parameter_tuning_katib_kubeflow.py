import kfp
from kfp import components, dsl

# Define the search space for the hyperparameters
hyperparameter_spec = {
    "parameters": [
        {
            "name": "hidden_size",
            "parameterType": "int",
            "feasibleSpace": {"min": "100", "max": "1000"},
        },
        {
            "name": "learning_rate",
            "parameterType": "double",
            "feasibleSpace": {"min": "0.0001", "max": "0.1"},
        },
    ]
}


# Define the Katib experiment
@dsl.pipeline(
    name="Hyperparameter tuning",
    description="Example of using Katib for hyperparameter tuning",
)
def hyperparameter_tuning_pipeline(
    hidden_size: int, learning_rate: float, model_dir: str
):
    # Define the training and evaluation steps
    train_op = components.load_component_from_url(
        "https://raw.githubusercontent.com/kubeflow/pipelines/master/components/kubeflow/tf-train/component.yaml"
    )
    eval_op = components.load_component_from_url(
        "https://raw.githubusercontent.com/kubeflow/pipelines/master/components/kubeflow/tf-job-bare/component.yaml"
    )

    # Set up the Katib optimizer
    optimizer = dsl.HyperparamOptimizer(
        hyperparameter_spec=hyperparameter_spec,
        algorithm="random",
        maxTrialCount=5,
        maxParallelTrialCount=1,
    )

    # Run the training and evaluation steps with the optimized hyperparameters
    train_task = train_op(
        model_dir=model_dir, hidden_size=hidden_size, learning_rate=learning_rate
    )
    eval_task = eval_op(model_dir=model_dir)

    eval_task.after(train_task)


# Compile the pipeline
pipeline_func = hyperparameter_tuning_pipeline
pipeline_filename = pipeline_func.__name__ + ".pipeline.zip"
compiler.Compiler().compile(pipeline_func, pipeline_filename)

# Create the Kubeflow Pipeline client
client = kfp.Client()

# Create a new experiment
experiment = client.create_experiment(name="Katib example")

# Create a new run
run_name = "Hyperparameter tuning run"
run = client.run_pipeline(experiment.id, run_name, pipeline_filename)
