import mlflow
import mlflow.pytorch
import hashlib
import mysql.connector


def hash_file(filepath):
    """Generate SHA-1 hash of a file."""
    with open(filepath, "rb") as f:
        bytes = f.read()
        file_hash = hashlib.sha1(bytes).hexdigest()
    return file_hash


def log_model_to_mlflow(model, model_path, artifact_uri):
    """
    Log PyTorch model to MLflow and return saved model path.

    :param model: PyTorch model
    :param model_path: path to save the model
    :param artifact_uri: MLflow server URI
    :return: path to the saved model
    """
    mlflow.set_tracking_uri(artifact_uri)
    mlflow.start_run()
    mlflow.pytorch.log_model(model, model_path)
    mlflow.end_run()

    return f"{artifact_uri}/{mlflow.active_run().info.run_id}/artifacts/{model_path}"


def write_hash_to_mysql(database_config, model_hash):
    """
    Write model hash to a MySQL database.

    :param database_config: dictionary containing MySQL connection parameters
    :param model_hash: model's SHA-1 hash
    """
    connection = mysql.connector.connect(**database_config)
    cursor = connection.cursor()
    insert_query = "INSERT INTO model_hashes (hash) VALUES (%s)"
    cursor.execute(insert_query, (model_hash,))
    connection.commit()
    cursor.close()
    connection.close()


# Define your model (example PyTorch model)
import torch.nn as nn

model = nn.Linear(10, 10)

# Define MLflow server URI
MLFLOW_URI = "http://your_mlflow_server:5000"

# Log model to MLflow
saved_model_path = log_model_to_mlflow(model, "linear_model", MLFLOW_URI)

# Compute the hash for the saved model
model_hash = hash_file(saved_model_path)

# Database configuration
db_config = {
    "host": "localhost",
    "user": "your_user",
    "password": "your_password",
    "database": "your_database",
}

# Create a table (just once) to store model hashes
connection = mysql.connector.connect(**db_config)
cursor = connection.cursor()
cursor.execute(
    "CREATE TABLE IF NOT EXISTS model_hashes (id INT AUTO_INCREMENT PRIMARY KEY, hash VARCHAR(255))"
)
connection.commit()
cursor.close()
connection.close()

# Write model hash to MySQL database
write_hash_to_mysql(db_config, model_hash)
