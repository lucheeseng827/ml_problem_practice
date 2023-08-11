import boto3

# Initialize boto3 client
s3 = boto3.client("s3")

# List of predefined buckets
predefined_buckets = ["bucket1", "bucket2", "bucket3"]

# Bucket to set lifecycle configuration
target_bucket = "bucket1"  # Replace with the name of the target bucket

# Check if the bucket is in the predefined list
if target_bucket not in predefined_buckets:
    print(f"Bucket {target_bucket} is not in the predefined list.")
else:
    # Define lifecycle policy
    lifecycle_policy = {
        "Rules": [
            {
                "ID": "MyRule",
                "Status": "Enabled",  # Can be 'Enabled' or 'Disabled'
                "Prefix": "",  # Empty prefix means the rule will apply to all objects in the bucket
                "Transitions": [
                    {
                        # Archive to Glacier after 30 days
                        "Days": 30,
                        "StorageClass": "GLACIER",
                    },
                ],
                # Optionally add a rule to delete after a number of days
                "Expiration": {"Days": 365},
            }
        ]
    }

    # Set lifecycle policy
    s3.put_bucket_lifecycle_configuration(
        Bucket=target_bucket, LifecycleConfiguration=lifecycle_policy
    )
    print(f"Lifecycle policy set for {target_bucket}")
