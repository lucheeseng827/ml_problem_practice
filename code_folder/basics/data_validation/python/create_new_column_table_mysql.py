import pymysql

# Database connection parameters
HOST = "your_host"
USER = "your_user"
PASSWORD = "your_password"
DATABASE = "your_database"
TABLE = "your_table"

# Connect to the database
connection = pymysql.connect(host=HOST, user=USER, password=PASSWORD, database=DATABASE)

try:
    with connection.cursor() as cursor:
        # Add a new column 'lastupdated' of type DATETIME
        sql = f"ALTER TABLE {TABLE} ADD lastupdated DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"
        cursor.execute(sql)

    # Commit the changes
    connection.commit()

finally:
    connection.close()
