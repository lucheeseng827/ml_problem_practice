import sendgrid

sg = sendgrid.SendGridAPIClient(apikey='your-sendgrid-api-key')

def send_email(to, subject, body):
    from_email = Email("test@example.com")
    to_email = Email(to)
    content = Content("text/plain", body)
    mail = Mail(from_email, subject, to_email, content)
    response = sg.client.mail.send.post(request_body=mail.get())
    print(response.status_code)
    print(response.body)
    print(response.headers)

send_email("test@example.com", "Hello, World!", "This is a test email sent using the SendGrid library.")
