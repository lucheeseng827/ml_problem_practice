import json

import boto3


def lambda_handler(event, context):
    # Extract parameters from the event
    notebook_name = event["notebook_name"]
    instance_type = event["instance_type"]
    dataset_url = event["dataset_url"]

    # Create a SageMaker client
    sagemaker_client = boto3.client("sagemaker")

    try:
        # Create the notebook instance
        response = sagemaker_client.create_notebook_instance(
            NotebookInstanceName=notebook_name,
            InstanceType=instance_type,
            RoleArn="arn:aws:iam::123456789012:role/SageMakerRole",  # Replace with your SageMaker role ARN
        )

        # Wait for the notebook instance to be in 'InService' status
        sagemaker_client.get_waiter("notebook_instance_in_service").wait(
            NotebookInstanceName=notebook_name
        )

        # Get the notebook instance details
        response = sagemaker_client.describe_notebook_instance(
            NotebookInstanceName=notebook_name
        )

        # Extract the SageMaker execution role from the response
        execution_role = response["RoleArn"]

        # Create an S3 client
        s3_client = boto3.client("s3")

        # Download the dataset from the provided URL
        dataset_filename = dataset_url.split("/")[-1]
        s3_client.download_file(dataset_url, "/tmp/{}".format(dataset_filename))

        # Upload the dataset to the notebook instance
        s3_client.upload_file(
            "/tmp/{}".format(dataset_filename),
            response["DefaultCodeRepository"],
            dataset_filename,
        )

        return {
            "statusCode": 200,
            "body": json.dumps(
                "SageMaker notebook created and dataset downloaded successfully!"
            ),
        }
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps("Error: {}".format(str(e)))}
