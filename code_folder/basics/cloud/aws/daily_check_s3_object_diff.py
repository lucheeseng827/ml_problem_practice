from datetime import datetime, timedelta

import boto3

# Set your S3 Inventory report name and the list of buckets to monitor
report_name = "your-inventory-report-name"
buckets_to_monitor = ["bucket-1", "bucket-2"]


def lambda_handler(event, context):
    # Create an S3 client
    s3_client = boto3.client("s3")

    # Get the start and end dates for the inventory report
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=1)

    # Retrieve the S3 Inventory report for the previous day
    for bucket_name in buckets_to_monitor:
        response = s3_client.get_bucket_inventory_configuration(
            Bucket=bucket_name, Id=report_name
        )
        inventory_destination = response["Destination"]
        inventory_format = inventory_destination["S3BucketDestination"]["Format"]
        inventory_bucket = inventory_destination["S3BucketDestination"]["Bucket"]
        inventory_prefix = inventory_destination["S3BucketDestination"]["Prefix"]
        inventory_frequency = response["Schedule"]["Frequency"]

        report_date = start_date.strftime("%Y-%m-%d")
        report_prefix = f"{inventory_prefix}{report_date}/"

        report_request = {
            "Bucket": bucket_name,
            "Format": inventory_format,
            "Prefix": report_prefix,
            "StartDate": start_date,
            "EndDate": end_date,
        }

        # Retrieve the S3 Inventory report for the previous day
        response = s3_client.generate_bucket_inventory(
            Bucket=bucket_name, Id=report_name, InventoryConfiguration=report_request
        )

        # Calculate the total size and number of objects created within the last 24 hours
        total_size = 0
        total_objects = 0
        for record in response["Inventory"]["Contents"]:
            if record["LastModifiedDate"].date() >= start_date:
                total_size += record["Size"]
                total_objects += 1

        # Send a summary email using Amazon SES or any other email service of your choice
        # Include the total size and number of objects created in the last 24 hours
        message = f"Total size of objects created in {bucket_name} on {report_date}: {total_size} bytes\n"
        message += f"Total number of objects created in {bucket_name} on {report_date}: {total_objects}"
        # Send the message using SES or any other email service of your choice
