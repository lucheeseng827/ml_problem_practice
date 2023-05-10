import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("insurance.csv")

# Let's assume that these are the significant features according to some domain knowledge or feature selection technique
significant_features = ["age", "bmi", "smoker"]

X = df[significant_features]
y = df["charges"]

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Selection
selector = SelectKBest(f_regression, k=2)
X_train_transformed = selector.fit_transform(X_train, y_train)
X_test_transformed = selector.transform(X_test)

# Create a linear regression object
lm = LinearRegression()

# Train the model using the training sets
lm.fit(X_train_transformed, y_train)

# Make predictions using the testing set
y_pred = lm.predict(X_test_transformed)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
