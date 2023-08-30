import pyodbc
from azure.cosmos import CosmosClient
from azure.servicebus import ServiceBusClient, ServiceBusMessage
from azure.storage.blob import BlobClient


# 1. Fetch data from Azure SQL
def fetch_data_from_sql():
    connection_string = "YOUR_AZURE_SQL_CONNECTION_STRING"
    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM YourTable")
    rows = cursor.fetchall()
    conn.close()
    return rows


# Alternatively, you can fetch data from CosmosDB
def fetch_data_from_cosmosdb():
    cosmos_endpoint = "YOUR_COSMOSDB_ENDPOINT"
    cosmos_key = "YOUR_COSMOSDB_KEY"
    database_name = "YOUR_DATABASE_NAME"
    container_name = "YOUR_CONTAINER_NAME"

    client = CosmosClient(cosmos_endpoint, cosmos_key)
    container = client.get_database_client(database_name).get_container_client(
        container_name
    )
    items = list(
        container.read_all_items(max_item_count=10)
    )  # adjust max_item_count accordingly
    return items


# 2. Send data to Azure Service Bus
def send_to_service_bus(data):
    service_bus_conn_str = "YOUR_SERVICE_BUS_CONNECTION_STRING"
    queue_name = "YOUR_QUEUE_NAME"

    with ServiceBusClient.from_connection_string(service_bus_conn_str) as client:
        with client.get_queue_sender(queue_name) as sender:
            for item in data:
                message = ServiceBusMessage(str(item))
                sender.send_messages(message)


# 3. Read from Azure Service Bus and write to Blob Storage
def consume_from_service_bus_and_write_to_blob():
    blob_conn_str = "YOUR_BLOB_CONNECTION_STRING"
    blob_container_name = "YOUR_CONTAINER_NAME"
    blob_name = "YOUR_BLOB_NAME"
    service_bus_conn_str = "YOUR_SERVICE_BUS_CONNECTION_STRING"
    queue_name = "YOUR_QUEUE_NAME"

    blob_client = BlobClient.from_connection_string(
        blob_conn_str, blob_container_name, blob_name
    )

    with ServiceBusClient.from_connection_string(service_bus_conn_str) as client:
        with client.get_queue_receiver(queue_name) as receiver:
            for message in receiver:
                blob_client.upload_blob(message)
                message.complete()


# Main execution
if __name__ == "__main__":
    # Fetch data from Azure SQL or CosmosDB
    data = fetch_data_from_sql()
    # data = fetch_data_from_cosmosdb()

    # Send the fetched data to Service Bus
    send_to_service_bus(data)

    # Consume data from Service Bus and write to Blob Storage
    consume_from_service_bus_and_write_to_blob()
