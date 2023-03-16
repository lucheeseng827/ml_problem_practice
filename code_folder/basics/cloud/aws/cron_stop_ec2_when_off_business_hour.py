import boto3
import os
from datetime import datetime

# Update your timezone, business hours start and end time accordingly
TIMEZONE = "America/New_York"
BUSINESS_START_HOUR = 9  # 9 AM
BUSINESS_END_HOUR = 17   # 5 PM

# Environment variables
INSTANCE_IDS = os.environ["INSTANCE_IDS"].split(",")

ec2 = boto3.client("ec2")


def lambda_handler(event, context):
    now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Current time: {now}")

    current_hour = datetime.now().astimezone().hour
    is_business_hour = BUSINESS_START_HOUR <= current_hour < BUSINESS_END_HOUR

    if is_business_hour:
        start_ec2_instances(INSTANCE_IDS)
    else:
        stop_ec2_instances(INSTANCE_IDS)


def start_ec2_instances(instance_ids):
    print(f"Starting EC2 instances: {instance_ids}")
    ec2.start_instances(InstanceIds=instance_ids)


def stop_ec2_instances(instance_ids):
    print(f"Stopping EC2 instances: {instance_ids}")
    ec2.stop_instances(InstanceIds=instance_ids)
