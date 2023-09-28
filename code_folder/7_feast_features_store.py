import pandas as pd
from feast import FeatureStore
from sklearn.ensemble import RandomForestClassifier

# Initialize a feature store
fs = FeatureStore(repo_path="path_to_your_repo/")

# Define an entity dataframe, which is a dataframe with timestamps and entity IDs
entity_df = pd.DataFrame.from_dict(
    {
        "driver_id": [1001, 1002, 1003],
        "event_timestamp": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]),
    }
)

# Fetch training data from Feast
training_df = fs.get_historical_features(
    entity_df=entity_df,
    features=["driver_hourly_stats:avg_rating", "driver_hourly_stats:trips_completed"],
).to_df()

# Split features and target variable (assuming 'avg_rating' is target)
X_train = training_df[["driver_hourly_stats:trips_completed"]]
y_train = training_df["driver_hourly_stats:avg_rating"]

# Train a model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
