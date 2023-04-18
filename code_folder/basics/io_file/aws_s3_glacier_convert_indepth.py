import json
import os
import sys
import time

import boto3

# instantiate boto3
s3 = boto3.client("s3")
glacier = boto3.client("glacier")


# create a list of s3 glacier object and state of it
def get_s3_glacier_objects():
    s3_glacier_objects = []
    response = s3.list_objects_v2(Bucket=os.environ["S3_BUCKET"])
    for obj in response["Contents"]:
        if obj["Size"] > 0:
            s3_glacier_objects.append(obj["Key"])
    return s3_glacier_objects


# change s3 glacier object state to s3 object
def change_s3_glacier_object_state(s3_glacier_objects):
    for s3_glacier_object in s3_glacier_objects:
        response = glacier.describe_job(
            VaultName=os.environ["GLACIER_VAULT"], JobId=s3_glacier_object.split("/")[0]
        )
        if response["Completed"]:
            s3.copy_object(
                Bucket=os.environ["S3_BUCKET"],
                CopySource=os.environ["S3_BUCKET"] + "/" + s3_glacier_object,
                Key=s3_glacier_object.split("/")[1],
            )
            glacier.delete_archive_from_job(
                VaultName=os.environ["GLACIER_VAULT"],
                JobId=s3_glacier_object.split("/")[0],
                ArchiveId=s3_glacier_object.split("/")[1],
            )
            s3.delete_object(Bucket=os.environ["S3_BUCKET"], Key=s3_glacier_object)


# main function
def main():
    s3_glacier_objects = get_s3_glacier_objects()
    change_s3_glacier_object_state(s3_glacier_objects)


if __name__ == "__main__":
    main()
