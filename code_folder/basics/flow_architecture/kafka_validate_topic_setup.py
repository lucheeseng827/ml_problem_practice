from confluent_kafka import Consumer, KafkaError


def check_kafka_setup(broker_url, group_id):
    conf = {
        "bootstrap.servers": broker_url,
        "group.id": group_id,
        "session.timeout.ms": 6000,
        "auto.offset.reset": "earliest",
    }

    consumer = Consumer(conf)

    # Wait up to 10s for the cluster metadata
    broker_metadata = consumer.list_topics(timeout=10)
    if broker_metadata is None:
        print("Failed to fetch broker metadata.")
        return

    topics = broker_metadata.topics
    for topic_name, topic_metadata in topics.items():
        print(f"Topic name: {topic_name}")
        print(f"Topic error: {topic_metadata.error}")

        partitions = topic_metadata.partitions
        print(f"Number of partitions: {len(partitions)}")

        for partition_id, partition_metadata in partitions.items():
            print(f"  Partition ID: {partition_id}")
            print(f"  Partition error: {partition_metadata.error}")
            print(f"  Partition leader: {partition_metadata.leader}")
            print(f"  Replicas: {partition_metadata.replicas}")
            print(f"  ISR: {partition_metadata.isrs}")

    consumer.close()


if __name__ == "__main__":
    broker = "YOUR_BROKER_URL"  # for example, 'localhost:9092'
    group = "YOUR_GROUP_ID"
    check_kafka_setup(broker, group)
