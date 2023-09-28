import json

import boto3
import requests
from requests.auth import HTTPBasicAuth

# S3 Setup
S3_BUCKET = "your_bucket_name"
s3 = boto3.client("s3")

# Elasticsearch setup
ES_HOST = "your_elasticsearch_host"
INDEX_NAME = "your_index_name"
ES_USERNAME = "your_username"  # if you have basic auth enabled for Elasticsearch
ES_PASSWORD = "your_password"  # if you have basic auth enabled for Elasticsearch
HEADERS = {"Content-Type": "application/json"}


def fetch_from_s3(filename):
    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=filename)
        data = response["Body"].read().decode("utf-8")
        return json.loads(data)
    except Exception as e:
        print(f"Error fetching from S3: {e}")
        return None


def index_to_elasticsearch(data):
    for record in data:
        # Assuming each record in your data contains an 'id' that you want to use as the Elasticsearch doc ID
        doc_id = record["id"]
        url = f"{ES_HOST}/{INDEX_NAME}/_doc/{doc_id}"
        response = requests.post(
            url,
            auth=HTTPBasicAuth(ES_USERNAME, ES_PASSWORD),
            headers=HEADERS,
            json=record,
        )
        if response.status_code != 201:  # 201 is HTTP status code for "Created"
            print(f"Error indexing record {doc_id}: {response.text}")


def main():
    # Assuming files in S3 are named like "batch_0.json", "batch_1.json" and so on.
    page = 0
    while True:
        filename = f"batch_{page}.json"
        data = fetch_from_s3(filename)

        if not data:
            break

        index_to_elasticsearch(data)
        page += 1


if __name__ == "__main__":
    main()
