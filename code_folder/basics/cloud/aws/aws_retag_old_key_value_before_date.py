from datetime import datetime, timedelta

import boto3

# Set up Boto3 client for the relevant AWS service
client = boto3.client("ec2")

# Define the tag key and value to match
old_tag_key = "OldTag"
old_tag_value = "ToBeReplaced"

# Define the new tag key and value
new_tag_key = "NewTag"
new_tag_value = "Replaced"

# Define the cutoff date for filtering resources
cutoff_date = datetime(2022, 1, 1)

# Retrieve all resources with the old tag
response = client.describe_instances(
    Filters=[
        {
            "Name": f"tag:{old_tag_key}",
            "Values": [
                old_tag_value,
            ],
        },
    ]
)

# Iterate over each resource and check if it was created before the cutoff date
for reservation in response["Reservations"]:
    for instance in reservation["Instances"]:
        if instance["LaunchTime"] < cutoff_date:
            # Create a new tag set with the old tag removed and the new tag added
            tags = [
                {"Key": t["Key"], "Value": t["Value"]}
                for t in instance["Tags"]
                if t["Key"] != old_tag_key
            ]
            tags.append({"Key": new_tag_key, "Value": new_tag_value})

            # Update the tags for the instance
            response = client.create_tags(
                Resources=[
                    instance["InstanceId"],
                ],
                Tags=tags,
            )

            print(f"Updated tags for instance {instance['InstanceId']}: {tags}")
