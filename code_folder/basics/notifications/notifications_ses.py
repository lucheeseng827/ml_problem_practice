import boto3

# Set the AWS access keys and the region
AWS_ACCESS_KEY_ID = "your-access-key-id"
AWS_SECRET_ACCESS_KEY = "your-secret-access-key"
REGION = "us-east-1"

# Create an SES client
client = boto3.client(
  "ses",
  aws_access_key_id=AWS_ACCESS_KEY_ID,
  aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
  region_name=REGION
)

# Set the email parameters
subject = "Notification from Python"
body = "This is a notification sent from a Python script."
to_address = "recipient@example.com"
from_address = "sender@example.com"

# Send the email
response = client.send_email(
  Destination={
    "ToAddresses": [
      to_address
    ]
  },
  Message={
    "Body": {
      "Text": {
        "Charset": "UTF-8",
        "Data": body
      }
    },
    "Subject": {
      "Charset": "UTF-8",
      "Data": subject
    }
  },
  Source=from_address
)

# Check the status code of the response
if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
  print("Email sent successfully.")
else:
  print("Error sending email:", response)
