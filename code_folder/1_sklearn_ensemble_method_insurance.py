import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('insurance.csv')

# Let's say we have a binary target variable 'claim' 
# which indicates whether a claim was filed (1) or not (0)
y = df['claim']

# The remaining columns are our features
X = df.drop('claim', axis=1)

# Convert categorical variables into numeric variables
label_encoder = LabelEncoder()
for col in X.select_dtypes(include=['object']).columns:
    X[col] = label_encoder.fit_transform(X[col])

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models
model1 = LogisticRegression(random_state=1)
model2 = DecisionTreeClassifier(random_state=1)
model3 = GaussianNB()

# Create the ensemble model
ensemble = VotingClassifier(estimators=[('lr', model1), ('dt', model2), ('gnb', model3)], voting='hard')

# Train the ensemble model
ensemble.fit(X_train, y_train)

# Make predictions
y_pred = ensemble.predict(X_test)

# Calculate and print accuracy
print('Accuracy:', accuracy_score(y_test, y_pred))
