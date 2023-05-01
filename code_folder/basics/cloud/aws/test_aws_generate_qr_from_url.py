import json

import boto3
from lambda_function import lambda_handler
from moto import mock_s3

s3 = boto3.client("s3")


@mock_s3
def test_lambda_handler():
    # Create a mock S3 bucket
    bucket_name = "my-bucket"
    s3.create_bucket(Bucket=bucket_name)

    # Define the input event
    event = {"url": "https://example.com"}

    # Call the Lambda function
    response = lambda_handler(event, None)

    # Check that the response is correct
    assert response["statusCode"] == 200
    assert response["body"] == "QR code image uploaded to S3"

    # Check that the image was uploaded to S3
    objects = s3.list_objects(Bucket=bucket_name)
    keys = [obj["Key"] for obj in objects.get("Contents", [])]
    assert len(keys) == 1
    assert keys[0].startswith("qrcodes/")
    assert keys[0].endswith(".png")

    # Check that the image can be downloaded from S3
    object_body = s3.get_object(Bucket=bucket_name, Key=keys[0])["Body"].read()
    assert len(object_body) > 0
