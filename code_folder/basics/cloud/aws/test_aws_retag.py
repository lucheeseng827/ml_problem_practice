import unittest
from datetime import datetime

import boto3
from moto import mock_ec2
from my_module import retag_resources


class TestRetagResources(unittest.TestCase):
    @mock_ec2
    def test_retag_instances(self):
        # Set up a mock EC2 client
        ec2 = boto3.client("ec2", region_name="us-east-1")

        # Create a test instance with the old tag
        instance = ec2.run_instances(
            ImageId="ami-0123456789abcdef0",
            MinCount=1,
            MaxCount=1,
            TagSpecifications=[
                {
                    "ResourceType": "instance",
                    "Tags": [{"Key": "OldTag", "Value": "ToBeReplaced"}],
                }
            ],
        )["Instances"][0]

        # Retag the instance using the function being tested
        retag_resources(
            ec2, "OldTag", "ToBeReplaced", "NewTag", "Replaced", datetime(2022, 1, 1)
        )

        # Retrieve the updated tags for the instance
        response = ec2.describe_instances(InstanceIds=[instance["InstanceId"]])
        tags = {
            t["Key"]: t["Value"]
            for t in response["Reservations"][0]["Instances"][0]["Tags"]
        }

        # Check that the instance was properly retagged
        self.assertEqual(tags["NewTag"], "Replaced")
        self.assertNotIn("OldTag", tags)


if __name__ == "__main__":
    unittest.main()
