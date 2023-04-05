from twilio.rest import Client

# Your Account SID and Auth Token from twilio.com/console
account_sid = "your-account-sid"
auth_token = "your-auth-token"
client = Client(account_sid, auth_token)

message = client.messages.create(
    body="Hello there!", from_="your-twilio-number", to="recipient-number"
)

print(message.sid)
