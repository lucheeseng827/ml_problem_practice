import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('insurance.csv')

# Convert categorical variables into numeric variables
label_encoder = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = label_encoder.fit_transform(df[col])

# Create the correlation matrix
corr_matrix = df.corr()

# Visualize the correlation matrix using seaborn's heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Here we assume 'charges' is the target variable and the rest are the features
# If 'charges' is not the target variable, replace it with the actual target variable
X = df.drop('charges', axis=1)
y = df['charges']

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression object
lm = LinearRegression()

# Train the model using the training sets
lm.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = lm.predict(X_test)

# The mean squared error
print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
