# First, you will need to set up authorization credentials for the Google API. You can do this by following the instructions here:
# https://developers.google.com/gmail/api/quickstart/python

# Then, install the required libraries:
# !pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client

# Import the necessary libraries
import base64
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

# Set the credentials object with your authorization credentials
creds = Credentials.from_authorized_user_info()

# Use the `build` function from the `googleapiclient.discovery` library to create a service object for the Gmail API
service = build("gmail", "v1", credentials=creds)

# Set the file ID for the file in Google Drive that you want to attach to the email
file_id = "FILE_ID_GOES_HERE"

# Use the `files().get()` method from the Drive API to retrieve the file
drive_service = build("drive", "v3", credentials=creds)
file = drive_service.files().get(fileId=file_id, fields="*").execute()

# Set the necessary parameters for the email
to = "RECIPIENT_EMAIL_ADDRESS_GOES_HERE"
subject = "EMAIL_SUBJECT_GOES_HERE"
body = "EMAIL_BODY_GOES_HERE"

# Create a message object with the necessary parameters
message = MIMEMultipart()
message["to"] = to
message["subject"] = subject

# Add the body of the email to the message
message.attach(MIMEText(body))

# Add the file as an attachment to the message
attachment = MIMEApplication(file["content"], _subtype=file["mimeType"])
attachment.add_header("Content-Disposition", "attachment", filename=file["name"])
message.attach(attachment)

# Use the `messages().send()` method to send the message
message = {"raw": base64.urlsafe_b64encode(message.as_bytes()).decode()}
send_message = service.users().messages().send(userId="me", body=message).execute()

print(f'sent message to {to} Message Id: {send_message["id"]}')
