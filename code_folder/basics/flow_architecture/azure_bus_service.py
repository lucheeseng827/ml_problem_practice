# pip install azure-servicebus
from azure.servicebus import Message, ServiceBusClient

# Replace YOUR_CONNECTION_STRING with the connection string for your Azure Service Bus namespace
servicebus_client = ServiceBusClient.from_connection_string("YOUR_CONNECTION_STRING")


queue_client = servicebus_client.get_queue_client("my-queue")


message = Message("Hello, World!")
queue_client.send(message)

with queue_client.get_receiver() as queue_receiver:
    messages = queue_receiver.fetch_next(timeout=5)
    for message in messages:
        print(message)
        message.complete()

# # Close the client

# https://docs.microsoft.com/en-us/azure/service-bus-messaging/service-bus-python-how-to-use-queues
