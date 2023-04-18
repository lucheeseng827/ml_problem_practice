import unittest

import boto3
from moto import mock_s3
from staleness_lambda import lambda_handler


class TestStalenessLambda(unittest.TestCase):
    @mock_s3
    def test_same_md5_hash_updates_metadata(self):
        # Set up mock S3 buckets and objects
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="source-bucket")
        s3.create_bucket(Bucket="destination-bucket")
        s3.put_object(Bucket="source-bucket", Key="test.txt", Body="Hello, World!")
        s3.put_object(Bucket="destination-bucket", Key="test.txt", Body="Hello, World!")

        # Create a mock S3 event
        event = {
            "Records": [
                {
                    "s3": {
                        "bucket": {"name": "source-bucket"},
                        "object": {"key": "test.txt"},
                    }
                }
            ]
        }

        # Call the Lambda function
        lambda_handler(event, None)

        # Check if the last_updated metadata was updated
        destination_response = s3.head_object(
            Bucket="destination-bucket", Key="test.txt"
        )
        self.assertIn("last_updated", destination_response["Metadata"])

    # Add more test methods for different scenarios
    # TODO: Add a test for when the MD5 hashes are different


if __name__ == "__main__":
    unittest.main()
