import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data and split into training and testing sets
data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=42
)

# Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

# Log the model with MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("my_experiment")

with mlflow.start_run():
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")
