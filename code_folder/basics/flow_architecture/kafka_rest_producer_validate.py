import json

import requests
from confluent_kafka import Producer


def send_request_to_endpoint(url, payload):
    headers = {
        "Content-Type": "application/json",
    }
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    if response.status_code != 200:
        print(f"Failed to get a response. Status code: {response.status_code}")
        return None
    return response.json()


def produce_to_topic(broker_url, topic_name, message):
    producer_conf = {
        "bootstrap.servers": broker_url,
    }
    producer = Producer(producer_conf)
    producer.produce(topic_name, json.dumps(message))
    producer.flush()


if __name__ == "__main__":
    # Kafka configuration
    broker = "YOUR_BROKER_URL"
    topic = "YOUR_TOPIC_NAME"

    # Endpoint configuration
    endpoint_url = "YOUR_ENDPOINT_URL"
    payload = {
        "key1": "value1",
        "key2": "value2",
        # Add more keys as necessary
    }

    response_data = send_request_to_endpoint(endpoint_url, payload)
    if response_data:
        produce_to_topic(broker, topic, response_data)
    else:
        print("Failed to send data to Kafka.")
