from confluent_kafka import Consumer, KafkaError, Producer


def topic_exists(broker_url, group_id, topic_name, expected_partitions=None):
    conf = {
        "bootstrap.servers": broker_url,
        "group.id": group_id,
        "session.timeout.ms": 6000,
        "auto.offset.reset": "earliest",
    }

    consumer = Consumer(conf)
    broker_metadata = consumer.list_topics(timeout=10)

    if broker_metadata is None:
        print("Failed to fetch broker metadata.")
        return False

    topics = broker_metadata.topics
    if topic_name not in topics:
        print(f"Topic {topic_name} does not exist.")
        return False

    if expected_partitions:
        partitions = topics[topic_name].partitions
        if len(partitions) != expected_partitions:
            print(
                f"Topic {topic_name} does not have the expected number of partitions."
            )
            return False

    # Add more validation checks as required

    consumer.close()
    return True


def produce_to_topic(broker_url, topic_name, message):
    producer_conf = {
        "bootstrap.servers": broker_url,
    }

    producer = Producer(producer_conf)
    producer.produce(topic_name, message)
    producer.flush()


if __name__ == "__main__":
    broker = "YOUR_BROKER_URL"  # e.g., 'localhost:9092'
    group = "YOUR_GROUP_ID"
    topic = "YOUR_TOPIC_NAME"
    expected_partitions_count = 3

    if topic_exists(broker, group, topic, expected_partitions_count):
        produce_to_topic(broker, topic, "Your message here")
    else:
        print(f"Failed to validate topic {topic}. Message was not sent.")
