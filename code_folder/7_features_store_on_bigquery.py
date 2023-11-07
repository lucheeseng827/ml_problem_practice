import os

from google.cloud import bigquery
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/your/service-account-file.json"

# Initialize a BigQuery client
client = bigquery.Client()

# Define your SQL query to fetch features
sql_query = """
    SELECT
        feature_1, feature_2, target_variable
    FROM
        `your_project_id.your_dataset.your_table`
    WHERE conditions_if_any
"""

# Execute the query and convert to pandas DataFrame
df = client.query(sql_query).to_dataframe()


# Split the data into training and testing sets
X = df[["feature_1", "feature_2"]]
y = df["target_variable"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Make predictions or evaluate the model
predictions = clf.predict(X_test)
