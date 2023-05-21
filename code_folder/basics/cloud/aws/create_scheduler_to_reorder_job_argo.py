from argo.workflows.client import Configuration, V1alpha1Api
from argo.workflows.config import load_kube_config

# Load the Kubernetes configuration
load_kube_config()

# Create a configuration object
configuration = Configuration()

# Create an instance of the Argo Workflow API client
api_client = V1alpha1Api(configuration)


def create_workflow_template(name, workflow_yaml):
    # Define the workflow template
    workflow_template = {
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "WorkflowTemplate",
        "metadata": {"name": name},
        "spec": {
            "templates": [
                {
                    "name": "job-template",
                    "container": {
                        "image": "your-image",
                        "command": ["your-command"],
                        "args": ["your-arguments"],
                    },
                }
            ],
            "workflows": [
                {
                    "name": "job-workflow",
                    "entrypoint": "job-template",
                    "templates": ["job-template"],
                }
            ],
        },
    }

    # Create the workflow template
    api_client.create_workflow_template(workflow_template)


def create_workflow_template(name, workflow_yaml):
    # Define the workflow template
    workflow_template = {
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "WorkflowTemplate",
        "metadata": {"name": name},
        "spec": {
            "templates": [
                {
                    "name": "job-template",
                    "container": {
                        "image": "your-image",
                        "command": ["your-command"],
                        "args": ["your-arguments"],
                    },
                }
            ],
            "workflows": [
                {
                    "name": "job-workflow",
                    "entrypoint": "job-template",
                    "templates": ["job-template"],
                }
            ],
        },
    }

    # Create the workflow template
    api_client.create_workflow_template(workflow_template)


create_workflow_template("my-workflow-template", "path/to/workflow.yaml")
