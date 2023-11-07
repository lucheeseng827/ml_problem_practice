
import jsonschema
from confluent_kafka import Consumer


def validate_json(json_data, schema):
    try:
        jsonschema.validate(instance=json_data, schema=schema)
        print("JSON data is valid against the schema.")
    except jsonschema.exceptions.ValidationError as e:
        print(f"JSON data validation error: {e}")


def consume_kafka_messages(bootstrap_servers, topic):
    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap_servers,
            "group.id": "test-group",
            "auto.offset.reset": "earliest",
        }
    )
    consumer.subscribe([topic])

    while True:
        message = consumer.poll(1.0)

        if message is None:
            continue
        if message.error():
            print(f"Kafka consumer error: {message.error()}")
            continue

        value = message.value().decode("utf-8")
        print(f"Consumed message: {value}")

        # Add your logic to test the consumed message as needed

        consumer.commit()


# JSON Schema for validation
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"},
        "email": {"type": "string", "format": "email"},
    },
    "required": ["name", "age", "email"],
}

# Test JSON data against the schema
json_data = {"name": "John Doe", "age": 30, "email": "johndoe@example.com"}
validate_json(json_data, schema)

# Test consuming messages from Kafka
bootstrap_servers = "localhost:9092"  # Replace with your Kafka bootstrap servers
topic = "test-topic"  # Replace with the Kafka topic to consume from
consume_kafka_messages(bootstrap_servers, topic)
