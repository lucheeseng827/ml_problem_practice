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
ES_USERNAME = "your_username"
ES_PASSWORD = "your_password"
HEADERS = {"Content-Type": "application/json"}

# File to store the last processed batch
LAST_PROCESSED_FILE = "last_processed.txt"


def fetch_from_s3(filename):
    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=filename)
        data = response["Body"].read().decode("utf-8")
        return json.loads(data)
    except Exception as e:
        print(f"Error fetching from S3: {e}")
        return None


def index_to_elasticsearch(data):
    success = True
    for record in data:
        doc_id = record.get("id", None)
        if not doc_id:
            print("Record does not have an ID. Skipping...")
            success = False
            continue

        url = f"{ES_HOST}/{INDEX_NAME}/_doc/{doc_id}"
        response = requests.post(
            url,
            auth=HTTPBasicAuth(ES_USERNAME, ES_PASSWORD),
            headers=HEADERS,
            json=record,
        )
        if response.status_code != 201:
            print(f"Error indexing record {doc_id}: {response.text}")
            success = False

    return success


def save_last_processed(batch_number):
    with open(LAST_PROCESSED_FILE, "w") as f:
        f.write(str(batch_number))


def get_last_processed():
    try:
        with open(LAST_PROCESSED_FILE, "r") as f:
            return int(f.read().strip())
    except FileNotFoundError:
        return -1


def main():
    page = get_last_processed() + 1
    while True:
        filename = f"batch_{page}.json"
        data = fetch_from_s3(filename)

        if not data:
            break

        if index_to_elasticsearch(data):
            save_last_processed(page)
        else:
            print(f"Error indexing data from {filename}. Stopping...")
            break

        page += 1


if __name__ == "__main__":
    main()
