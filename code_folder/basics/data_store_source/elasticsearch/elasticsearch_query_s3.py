import json
from time import sleep

import boto3
from elasticsearch import Elasticsearch
from elasticsearch import exceptions as es_exceptions

# Elasticsearch setup
ES_HOST = "your_elasticsearch_host"
INDEX_NAME = "your_index_name"
es = Elasticsearch(ES_HOST)

# S3 setup
S3_BUCKET = "your_bucket_name"
s3 = boto3.client("s3")

# Constants
BATCH_SIZE = 100
MAX_RETRIES = 3
DELAY_BETWEEN_RETRIES = 5  # in seconds


def save_to_s3(data, filename):
    try:
        s3.put_object(Bucket=S3_BUCKET, Key=filename, Body=json.dumps(data))
        print(f"Saved data to S3: {filename}")
    except Exception as e:
        print(f"Error saving to S3: {e}")


def query_elasticsearch(from_page):
    body = {
        "from": from_page * BATCH_SIZE,
        "size": BATCH_SIZE,
        "query": {"match_all": {}},
    }

    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = es.search(index=INDEX_NAME, body=body)
            hits = response["hits"]["hits"]
            if not hits:
                return None
            return hits
        except es_exceptions.ConnectionError:
            retries += 1
            print(f"Connection error, retry {retries}/{MAX_RETRIES}")
            sleep(DELAY_BETWEEN_RETRIES)

    raise Exception("Max retries reached for Elasticsearch connection.")


def main():
    page = 0
    while True:
        data = query_elasticsearch(page)
        if not data:
            break

        # Save this batch of data to S3
        filename = f"batch_{page}.json"
        save_to_s3(data, filename)

        page += 1


if __name__ == "__main__":
    main()
