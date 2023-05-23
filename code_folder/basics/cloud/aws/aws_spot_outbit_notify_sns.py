import boto3

def lambda_handler(event, context):
    instance_id = event['detail']['instance-id']
    region = event['region']

    sns_client = boto3.client('sns')
    topic_arn = '<your_SNS_topic_arn>'

    message = f"Spot instance {instance_id} in {region} has been outbid."

    response = sns_client.publish(
        TopicArn=topic_arn,
        Message=message,
        Subject='Spot Instance Outbid Notification'
    )

    print(f"Sent SNS notification: {response['MessageId']}")
