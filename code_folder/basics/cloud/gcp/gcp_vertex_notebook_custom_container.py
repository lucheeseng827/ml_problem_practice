import io
import os

from google.cloud import aiplatform

def create_notebook_instance(
    project_id, location, machine_type, no_public_ip, custom_container_image
):
    """Creates a GCP Vertex AI notebook instance on a non-public IP with a custom container.

    Args:
      project_id: The ID of your Google Cloud project.
      location: The region where you want to create the notebook instance.
      machine_type: The machine type of the notebook instance.
      no_public_ip: Whether or not the notebook instance should have a public IP address.
      custom_container_image: The name of the custom container image that you want to use.

    Returns:
      The created notebook instance.
    """

    # Create the AI Platform client object.
    client = aiplatform.gapic.NotebookServiceClient()

    # Create the notebook instance creation request object.
    request = aiplatform.gapic.types.CreateNotebookInstanceRequest()
    request.parent = f"projects/{project_id}/locations/{location}"
    request.notebook_instance.machine_type = machine_type
    request.notebook_instance.no_public_ip = no_public_ip
    request.notebook_instance.custom_container_image = custom_container_image

    # Create the notebook instance.
    response = client.create_notebook_instance(request)

    # Return the created notebook instance.
    return response.notebook_instance


if __name__ == "__main__":
    # Set the project ID and location.
    project_id = os.environ["GOOGLE_CLOUD_PROJECT"]
    location = "us-central1"

    # Set the machine type and custom container image.
    machine_type = "n1-standard-8"
    custom_container_image = "gcr.io/my-project/my-container-image"

    # Create the notebook instance.
    notebook_instance = create_notebook_instance(
        project_id, location, machine_type, True, custom_container_image
    )

    # Print the notebook instance name.
    print(notebook_instance.name)
