import mysql.connector
import pandas as pd

# Connect to MySQL
mysql_connection = mysql.connector.connect(
    host="localhost",
    user="your_username",
    password="your_password",
    database="your_database",
)


# Define the source data extraction
def extract_data():
    # Your code to extract data from the source
    # This can involve querying APIs, reading files, etc.
    # For simplicity, let's assume we have a pandas DataFrame as the source data
    source_data = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "customer_name": ["Alice", "Bob", "Charlie"],
            "email": ["alice@example.com", "bob@example.com", "charlie@example.com"],
        }
    )
    return source_data


# Transform the source data if needed
def transform_data(source_data):
    # Your code to transform the source data
    # This can involve data cleaning, filtering, mapping, etc.
    transformed_data = source_data.copy()
    transformed_data["customer_name"] = transformed_data["customer_name"].str.upper()
    return transformed_data


# Load data into MySQL
def load_data(target_data):
    cursor = mysql_connection.cursor()

    # Create a table if it doesn't exist
    create_table_query = """
    CREATE TABLE IF NOT EXISTS customers (
        customer_id INT PRIMARY KEY,
        customer_name VARCHAR(255),
        email VARCHAR(255)
    )
    """
    cursor.execute(create_table_query)

    # Insert data into the table
    insert_query = (
        "INSERT INTO customers (customer_id, customer_name, email) VALUES (%s, %s, %s)"
    )
    values = target_data.to_records(index=False).tolist()
    cursor.executemany(insert_query, values)

    # Commit the changes
    mysql_connection.commit()
    cursor.close()


# Run the Reverse ETL process
def run_reverse_etl():
    source_data = extract_data()
    transformed_data = transform_data(source_data)
    load_data(transformed_data)


# Execute the Reverse ETL process
run_reverse_etl()

# Close the MySQL connection
mysql_connection.close()
