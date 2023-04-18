import base64
import json
import os

from google.auth import compute_engine
from google.cloud import container_v1, storage


def build_and_push_golden_image(event, context):
    # Get environment variables
    gcs_bucket_name = os.environ["GCS_BUCKET_NAME"]
    gcs_object_name = os.environ["GCS_OBJECT_NAME"]
    dockerhub_username = os.environ["DOCKERHUB_USERNAME"]
    dockerhub_password = os.environ["DOCKERHUB_PASSWORD"]
    golden_image_name = os.environ["GOLDEN_IMAGE_NAME"]
    golden_image_version = os.environ["GOLDEN_IMAGE_VERSION"]

    # Create GCS client
    credentials = compute_engine.Credentials()
    storage_client = storage.Client(credentials=credentials)

    # Download Dockerfile from GCS bucket
    bucket = storage_client.bucket(gcs_bucket_name)
    blob = bucket.blob(gcs_object_name)
    dockerfile_content = blob.download_as_string()

    # Create GCS job for building golden image
    container_client = container_v1.ContainerClient()
    operation = container_client.create_build_operation(
        project_id="my-project",
        build=container_v1.Build(
            steps=[
                container_v1.BuildStep(
                    name="gcr.io/cloud-builders/docker",
                    args=[
                        "build",
                        "-t",
                        f"{dockerhub_username}/{golden_image_name}:{golden_image_version}",
                        "-",
                    ],
                    id="build-docker-image",
                    waitFor=[],
                    entrypoint="",
                    dir="/workspace",
                )
            ],
            options=container_v1.BuildOptions(
                env=[
                    container_v1.BuildOptions.EnvVariable(
                        name="DOCKERHUB_USERNAME", value=dockerhub_username
                    ),
                    container_v1.BuildOptions.EnvVariable(
                        name="DOCKERHUB_PASSWORD", value=dockerhub_password
                    ),
                ]
            ),
            source=container_v1.Source(
                storage_source=container_v1.StorageSource(
                    bucket=gcs_bucket_name,
                    object_=gcs_object_name,
                    generation=blob.generation,
                )
            ),
            timeout={"seconds": 600},
            logs_bucket=f"my-project-build-logs",
        ),
    )

    # Wait for GCS job to complete
    operation.result()

    # Tag and push golden image to Docker Hub
    os.system(
        f"docker tag {dockerhub_username}/{golden_image_name}:{golden_image_version} {dockerhub_username}/{golden_image_name}:latest"
    )
    os.system(f"docker push {dockerhub_username}/{golden_image_name}:latest")
