import unittest

import mlflow.sklearn
import numpy as np
import pandas as pd

LOCAL_MODEL_URI = "./mlruns/0/your-run-id/artifacts/model"


class TestMLflowModelLoadingFromLocalStorage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the model using MLflow
        cls.model = mlflow.sklearn.load_model(LOCAL_MODEL_URI)

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
