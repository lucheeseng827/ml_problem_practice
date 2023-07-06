from kafka import KafkaProducer
import yaml
import time

def load_config():
    with open('kafka_producer_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_message(name, metadata):
    timestamp = int(time.time())
    message = {
        'name': name,
        'timestamp': timestamp,
        'metadata': metadata
    }
    return message

def send_message(producer, topic, message):
    producer.send(topic, value=message.encode('utf-8'))
    producer.flush()

def main():
    config = load_config()
    kafka_servers = config['kafka_servers']
    topic = config['topic']
    name = config['name']
    metadata = config['metadata']

    producer = KafkaProducer(bootstrap_servers=kafka_servers)

    message = create_message(name, metadata)
    send_message(producer, topic, message)

    producer.close()

if __name__ == '__main__':
    main()
