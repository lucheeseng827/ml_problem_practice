import psycopg2

# Database connection parameters
HOST = "your_host"
USER = "your_user"
PASSWORD = "your_password"
DATABASE = "your_database"
TABLE = "your_table"

# Connect to the database
connection = psycopg2.connect(host=HOST, user=USER, password=PASSWORD, dbname=DATABASE)

try:
    with connection.cursor() as cursor:
        # Add a new column 'lastupdated' of type TIMESTAMP with a default value of the current timestamp
        sql = f"ALTER TABLE {TABLE} ADD COLUMN lastupdated TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        cursor.execute(sql)

    # Commit the changes
    connection.commit()

finally:
    connection.close()
