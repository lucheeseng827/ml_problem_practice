import os
import unittest
import mlflow.sklearn
import pandas as pd
import numpy as np
from google.cloud import storage

# Set environment variable for Google Cloud authentication
os.environ[
    "GOOGLE_APPLICATION_CREDENTIALS"
] = "path/to/your/google-cloud-service-account-key.json"

GCS_MODEL_URI = "gs://your-bucket-name/path/to/your/model"


class TestMLflowModelLoadingFromCloudStorage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Download the model from Google Cloud Storage
        storage_client = storage.Client()
        bucket_name, blob_path = cls.parse_gcs_uri(GCS_MODEL_URI)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        model_path = "/tmp/model"
        blob.download_to_filename(model_path)

        # Load the model using MLflow
        cls.model = mlflow.sklearn.load_model(model_path)

    @staticmethod
    def parse_gcs_uri(uri):
        """Parse a Google Cloud Storage URI."""
        parts = uri.split("/", 3)
        bucket_name = parts[2]
        blob_path = parts[3]
        return bucket_name, blob_path

    def test_model_loading_and_validation(self):
        # Prepare input data for validation
        input_data = pd.DataFrame({"feature_1": [6, 7, 8], "feature_2": [7, 8, 9]})

        # Make predictions with the loaded model
        y_pred = self.model.predict(input_data)

        # Assert the model's output
        expected_output = np.array([13, 15, 17])
        np.testing.assert_array_almost_equal(y_pred, expected_output, decimal=1)


if __name__ == "__main__":
    unittest.main()
