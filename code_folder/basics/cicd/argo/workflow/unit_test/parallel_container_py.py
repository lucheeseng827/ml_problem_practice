import time

import requests

# Define the URL of the Argo API server
api_url = "http://localhost:2746/api/v1"

# Define the name of the workflow and the namespace in which it will run
workflow_name = "my-parallel-workflow"
namespace = "default"

# Define the inputs for the workflow
inputs = {"message1": "Hello from container 1!", "message2": "Hello from container 2!"}

# Define the expected outputs of the workflow
expected_outputs = {
    "output1": "Hello from container 1!",
    "output2": "Hello from container 2!",
}


# Define a test function that will run the workflow and check the outputs
def test_workflow():
    # Create the workflow resource
    workflow = {
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "Workflow",
        "metadata": {"generateName": workflow_name + "-"},
        "spec": {
            "entrypoint": "parallel",
            "templates": [
                {
                    "name": "container1",
                    "container": {
                        "image": "my-container-image",
                        "command": [
                            "sh",
                            "-c",
                            "echo '{{inputs.parameters.message1}}'; sleep 5",
                        ],
                        "resources": {"limits": {"cpu": "1"}},
                    },
                    "outputs": {
                        "parameters": [
                            {
                                "name": "output1",
                                "valueFrom": "{‌{steps.container1.outputs.result}}",
                            }
                        ]
                    },
                },
                {
                    "name": "container2",
                    "container": {
                        "image": "my-container-image",
                        "command": [
                            "sh",
                            "-c",
                            "echo '{{inputs.parameters.message2}}'; sleep 5",
                        ],
                        "resources": {"limits": {"cpu": "1"}},
                    },
                    "outputs": {
                        "parameters": [
                            {
                                "name": "output2",
                                "valueFrom": "{‌{steps.container2.outputs.result}}",
                            }
                        ]
                    },
                },
                {
                    "name": "container2",
                    "container": {
                        "image": "my-container-image",
                        "command": [
                            "sh",
                            "-c",
                            "echo '{{inputs.parameters.message2}}'; sleep 5",
                        ],
                        "resources": {"limits": {"cpu": "1"}},
                    },
                    "outputs": {
                        "parameters": [
                            {
                                "name": "output2",
                                "valueFrom": "{‌{steps.container2.outputs.result}}",
                            }
                        ]
                    },
                },
                {
                    "name": "parallel",
                    "steps": [
                        {"name": "container1", "template": "container1"},
                        {"name": "container2", "template": "container2"},
                    ],
                },
            ],
        },
    }

    # Submit the workflow to the Argo API server
    response = requests.post(
        api_url + "/namespaces/" + namespace + "/workflows", json=workflow
    )

    # Check that the response is successful
    assert response.status_code == 201

    # Get the workflow ID from the response
    workflow_id = response.json()["metadata"]["uid"]

    # Wait for the workflow to complete
    workflow_completed = False
    while not workflow_completed:
        # Get the status of the workflow from the Argo API server
        response = requests.get(
            api_url + "/namespaces/" + namespace + "/workflows/" + workflow_id
        )

        # Check that the response is successful
        assert response.status_code == 200

        # Get the status of the workflow
        workflow_status = response.json()["status"]["phase"]

        # Check if the workflow has completed
        if workflow_status in ["Succeeded", "Failed", "Error"]:
            workflow_completed = True
        else:
            # Wait for 5 seconds before checking the status again
            time.sleep(5)

    # Check the outputs of the workflow
    response = requests.get(
        api_url + "/namespaces/" + namespace + "/workflows/" + workflow_id + "/outputs"
    )
    assert response.status_code == 200
    outputs = response.json()["result"]
    assert outputs == expected_outputs
