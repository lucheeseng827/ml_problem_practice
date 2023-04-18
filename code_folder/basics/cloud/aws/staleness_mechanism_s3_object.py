import datetime

import boto3


def lambda_handler(event, context):
    s3 = boto3.client("s3")

    # Get source bucket and key from the event
    source_bucket = event["Records"][0]["s3"]["bucket"]["name"]
    source_key = event["Records"][0]["s3"]["object"]["key"]

    # Define the destination bucket
    destination_bucket = "your-destination-bucket"

    # Get the MD5 hash of the source object
    source_response = s3.head_object(Bucket=source_bucket, Key=source_key)
    source_md5 = source_response["ETag"][1:-1]

    try:
        # Get the MD5 hash of the destination object
        destination_response = s3.head_object(Bucket=destination_bucket, Key=source_key)
        destination_md5 = destination_response["ETag"][1:-1]
    except Exception as e:
        print(f"Error getting destination object: {e}")
        return

    # If the MD5 hashes are the same, update the last_updated metadata
    if source_md5 == destination_md5:
        now = datetime.datetime.now().isoformat()
        s3.copy_object(
            CopySource={
                "Bucket": destination_bucket,
                "Key": source_key,
            },
            Bucket=destination_bucket,
            Key=source_key,
            MetadataDirective="REPLACE",
            Metadata={
                "last_updated": now,
            },
        )
        print(f"Updated last_updated metadata of {source_key} to {now}")
    else:
        print(f"MD5 hashes are different, not updating metadata")
