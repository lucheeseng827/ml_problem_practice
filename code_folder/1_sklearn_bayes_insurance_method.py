import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

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

# Create a Gaussian Naive Bayes object
gnb = GaussianNB()

# Train the model using the training sets
gnb.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = gnb.predict(X_test)

# Print the accuracy score and the confusion matrix
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
