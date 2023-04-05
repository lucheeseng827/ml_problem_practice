import sendgrid

sg = sendgrid.SendGridAPIClient(apikey="your-sendgrid-api-key")


class Email(object):
    def __init__(self, email):
        self.email = email

    def get(self):
        return self.email


class Content(object):
    def __init__(self, content_type, value):
        self.type = content_type
        self.value = value

    def get(self):
        return {"type": self.type, "value": self.value}


class Personalization(object):
    def __init__(self):
        self.to = []
        self.cc = []
        self.bcc = []
        self.subject = None
        self.headers = {}
        self.substitutions = {}
        self.custom_args = {}
        self.send_at = None

    def get(self):
        personalization = {}
        if len(self.to) > 0:
            personalization["to"] = [email.get() for email in self.to]
        if len(self.cc) > 0:
            personalization["cc"] = [email.get() for email in self.cc]
        if len(self.bcc) > 0:
            personalization["bcc"] = [email.get() for email in self.bcc]
        if self.subject is not None:
            personalization["subject"] = self.subject
        if len(self.headers) > 0:
            personalization["headers"] = self.headers
        if len(self.substitutions) > 0:
            personalization["substitutions"] = self.substitutions
        if len(self.custom_args) > 0:
            personalization["custom_args"] = self.custom_args
        if self.send_at is not None:
            personalization["send_at"] = self.send_at
        return personalization


class Mail(object):
    def __init__(self, from_email, subject, to_email, content):
        self.from_email = from_email
        self.subject = subject
        self.to_email = to_email
        self.content = content
        self.personalizations = []

    def get(self):
        mail = {}
        mail["from"] = self.from_email.get()
        mail["subject"] = self.subject
        mail["content"] = [self.content.get()]
        if len(self.personalizations) > 0:
            mail["personalizations"] = [p.get() for p in self.personalizations]
        else:
            personalization = Personalization()
            personalization.to.append(self.to_email)
            mail["personalizations"] = [personalization.get()]
        return mail


def send_email(to, subject, body):
    from_email = Email("test@example.com")
    to_email = Email(to)
    content = Content("text/plain", body)
    mail = Mail(from_email, subject, to_email, content)
    response = sg.client.mail.send.post(request_body=mail.get())
    print(response.status_code)
    print(response.body)
    print(response.headers)


send_email(
    "test@example.com",
    "Hello, World!",
    "This is a test email sent using the SendGrid library.",
)
