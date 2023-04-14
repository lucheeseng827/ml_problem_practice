# This code creates volumes and then deletes them
# The volumes are created with the tag "Name: foo"
# The code then iterates through the volumes and deletes them
# The code uses the describe_volumes function to retrieve the volumes
# The code uses the delete_volume function to delete the volumes

import boto3
import pprint


client = boto3.client("ec2")
response = client.describe_volumes(Filters=[{"Name": "tag:Name", "Values": ["foo"]}])

volumes = response["Volumes"]

for volume in volumes:
    print(f'Volume {volume["VolumeId"]} has state {volume["State"]}')

for volume in volumes:
    client.delete_volume(VolumeId=volume["VolumeId"])
