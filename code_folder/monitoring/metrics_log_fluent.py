import fluent.sender

# Set the Fluent Bit host and port
FLUENT_BIT_HOST = "localhost"
FLUENT_BIT_PORT = 24224

# Create a Fluent Bit sender
sender = fluent.sender.FluentSender("my_tag", host=FLUENT_BIT_HOST, port=FLUENT_BIT_PORT)

# Send a log message
sender.emit("key1", {"key2": "value2"})

# Flush the logs
sender.flush()
