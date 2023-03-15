import boto3

# Set the AWS access keys and the region
AWS_ACCESS_KEY_ID = "your-access-key-id"
AWS_SECRET_ACCESS_KEY = "your-secret-access-key"
AWS_REGION = "us-east-1"

# Create an SNS client
client = boto3.client(
    "sns",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

# Set the topic ARN and the message
topic_arn = "arn:aws:sns:us-east-1:123456789012:my-topic"
message = "This is a message from Python."

# Send the message
response = client.publish(
    TopicArn=topic_arn,
    Message=message
)

# Print the message ID
print("Message ID:", response["MessageId"])
