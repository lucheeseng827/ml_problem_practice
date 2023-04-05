import boto3
import time
from datetime import datetime, timedelta

# Function to fetch spot price history


def get_spot_price_history(ec2_client, instance_type, availability_zone, start_time):
    response = ec2_client.describe_spot_price_history(
        InstanceTypes=[instance_type],
        ProductDescriptions=["Linux/UNIX"],
        AvailabilityZone=availability_zone,
        StartTime=start_time,
    )
    return response["SpotPriceHistory"]


# Parameters
instance_type = "t2.micro"
availability_zone = "us-west-2a"
monitor_interval = 60  # seconds

# Create an EC2 client
ec2 = boto3.client("ec2")

# Monitor spot price history
while True:
    start_time = datetime.utcnow() - timedelta(minutes=10)
    spot_price_history = get_spot_price_history(
        ec2, instance_type, availability_zone, start_time
    )

    for price_record in spot_price_history:
        timestamp = price_record["Timestamp"]
        price = price_record["SpotPrice"]
        print(f"{timestamp}: ${price}")

    time.sleep(monitor_interval)
