import pandas as pd
from gensim.models import Word2Vec

# Assuming you have a list of documents obtained from Elasticsearch results
documents = [result["_source"]["field_name"] for result in results["hits"]["hits"]]

# Train the Word2Vec model
model = Word2Vec(documents, size=100, window=5, min_count=1)

# Get the word embeddings
embeddings = model.wv

# Store the embeddings in a secondary data store, such as a database or file system
# Option 1: Storing embeddings in a pandas DataFrame and saving to a CSV file
embedding_df = pd.DataFrame(embeddings.vectors, index=embeddings.index2word)
embedding_df.to_csv("word_embeddings.csv")

# Option 2: Storing embeddings in a MySQL database using PyMySQL
import pymysql

# Connect to the MySQL database
connection = pymysql.connect(
    host="localhost", user="your_username", password="your_password", db="your_database"
)

# Create a table to store the embeddings
table_name = "word_embeddings"
with connection.cursor() as cursor:
    create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} (word VARCHAR(255) PRIMARY KEY, embedding BLOB)"
    cursor.execute(create_table_query)

# Insert the embeddings into the database
with connection.cursor() as cursor:
    for word, embedding in zip(embeddings.index2word, embeddings.vectors):
        insert_query = f"INSERT INTO {table_name} (word, embedding) VALUES (%s, %s)"
        cursor.execute(insert_query, (word, embedding.tobytes()))

# Commit the changes and close the connection
connection.commit()
connection.close()
