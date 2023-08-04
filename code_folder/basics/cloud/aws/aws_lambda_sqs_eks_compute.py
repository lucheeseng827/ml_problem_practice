import os

import boto3


def lambda_handler(event, context):
    region_a_cluster_name = os.environ["REGION_A_CLUSTER_NAME"]
    region_b_cluster_name = os.environ["REGION_B_CLUSTER_NAME"]
    threshold_percent = int(os.environ["THRESHOLD_PERCENT"])
    sqs_queue_url = os.environ["SQS_QUEUE_URL"]

    eks_client = boto3.client("eks")
    sqs_client = boto3.client("sqs")

    # Check compute resource availability in Region A
    region_a_capacity = get_eks_capacity(region_a_cluster_name, eks_client)
    region_a_available_capacity = region_a_capacity * (1 - threshold_percent / 100)

    if region_a_available_capacity <= 0:
        send_sqs_message(
            sqs_client, sqs_queue_url, "Compute resources running out in Region A"
        )

    # Check compute resource availability in Region B
    region_b_capacity = get_eks_capacity(region_b_cluster_name, eks_client)
    region_b_available_capacity = region_b_capacity * (1 - threshold_percent / 100)

    if region_b_available_capacity <= 0:
        send_sqs_message(
            sqs_client, sqs_queue_url, "Compute resources running out in Region B"
        )


def get_eks_capacity(cluster_name, eks_client):
    response = eks_client.describe_fargate_profile(
        clusterName=cluster_name, fargateProfileName="default"
    )
    capacity = response["fargateProfile"]["selectors"][0]["capacityProvider"]["value"]
    return capacity


def send_sqs_message(sqs_client, queue_url, message_body):
    response = sqs_client.send_message(QueueUrl=queue_url, MessageBody=message_body)
    return response
