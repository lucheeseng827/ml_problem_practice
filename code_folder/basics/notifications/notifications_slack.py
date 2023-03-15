import requests

# Set the Slack API token and the channel ID
SLACK_TOKEN = "xoxb-your-token"
CHANNEL_ID = "C1234567890"

# Set the message content
message = {
    "text": "This is a message from a Python script."
}

# Send the message
response = requests.post(
    "https://slack.com/api/chat.postMessage",
    headers={
        "Authorization": f"Bearer {SLACK_TOKEN}"
    },
    json={
        "channel": CHANNEL_ID,
        "text": "This is a message from a Python script."
    }
)

# Check the status code of the response
if response.status_code == 200:
    print("Message sent successfully.")
else:
    print("Error sending message:", response.status_code)
