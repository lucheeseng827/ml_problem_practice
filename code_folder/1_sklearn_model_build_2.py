# Import necessary libraries
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the dataset
data = pd.read_csv(".\insurance.csv")

# Preprocessing
features = data.iloc[:, :-1]
target = data.iloc[:, -1]

# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["age", "bmi", "children"]),
        ("cat", OneHotEncoder(), ["sex", "smoker", "region"]),
    ]
)

# Split the dataset into training set and test set
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.2, random_state=0
)

# Create a pipeline
pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("regressor", LinearRegression())]
)

# Train the model
pipeline.fit(features_train, target_train)

# Make predictions
target_pred = pipeline.predict(features_test)
print(f"Prediction : {target_pred}")

# Calculate and print the mean squared error
mse = mean_squared_error(target_test, target_pred)
print(f"Mean Squared Error: {mse}")
