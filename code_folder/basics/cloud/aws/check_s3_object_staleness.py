import boto3
import datetime


def check_stale_objects(bucket, threshold):
    s3 = boto3.client("s3")
    now = datetime.datetime.now(datetime.timezone.utc)
    stale_objects = []

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket):
        for obj in page.get("Contents", []):
            age = now - obj["LastModified"]
            if age > threshold:
                stale_objects.append(obj["Key"])

    return stale_objects
