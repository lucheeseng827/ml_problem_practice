"""To send a personal message to a user, you can use the /sendMessage endpoint of the API, and specify the user's email address as the to parameter in the request.

For more information about the Microsoft Teams API and other ways to send notifications to Microsoft Teams, you can refer to the documentation: https://docs.microsoft.com/en-us/microsoftteams/platform/concepts/connectors/connectors-using
"""

import requests

# Set the webhook URL
webhook_url = "https://outlook.office.com/webhook/{webhook_id}/{webhook_key}/{webhook_secret}/{webhook_path}"

# Set the message content
message = {
  "title": "Message from Python",
  "text": "This is a message from a Python script."
}

# Send the message
response = requests.post(webhook_url, json=message)

# Check the status code of the response
if response.status_code == 200:
  print("Message sent successfully.")
else:
  print("Error sending message:", response.status_code)
