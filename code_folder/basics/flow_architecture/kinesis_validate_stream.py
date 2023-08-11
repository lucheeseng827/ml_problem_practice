import json

import boto3


class KinesisProducer:
    def __init__(self, stream_name):
        self.client = boto3.client("kinesis")
        self.stream_name = stream_name

    def validate_stream(self):
        # Check if stream exists
        try:
            response = self.client.describe_stream(StreamName=self.stream_name)
            if response["StreamDescription"]["StreamStatus"] not in [
                "CREATING",
                "UPDATING",
                "ACTIVE",
            ]:
                return False

            # Here, you can add more validations based on your needs like:
            # - Checking the number of shards (partitions in Kafka's terms)
            # - Verifying any specific tags or configurations

            return True
        except self.client.exceptions.ResourceNotFoundException:
            print(f"Stream {self.stream_name} does not exist.")
            return False

    def put_data(self, data):
        if not self.validate_stream():
            print("Invalid Stream. Exiting...")
            return

        payload = {
            "Data": json.dumps(data),
            "PartitionKey": "some_partition_key",  # Change this based on your logic
            "StreamName": self.stream_name,
        }
        response = self.client.put_record(**payload)
        print(
            f"Record {response['SequenceNumber']} pushed to shard {response['ShardId']}"
        )


if __name__ == "__main__":
    producer = KinesisProducer("YourStreamNameHere")

    sample_data = {"key1": "value1", "key2": "value2"}

    producer.put_data(sample_data)
