import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Load dataset
data = pd.read_csv("mobile-phone-data.csv")

# Set display option to show all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)  # Optional: Show all rows (be cautious with large DataFrames)

# Separate features and target variable
X = data.drop("price_range", axis=1)  # Features
y = data["price_range"]  # Target variable

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train Decision Tree model
dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=10, max_features=None,
                                  min_samples_leaf=5, min_samples_split=2, random_state=42)
start_time = time.time()
dt_model.fit(X_train, y_train)
training_time = time.time() - start_time


# Make predictions
start_time = time.time()
y_pred = dt_model.predict(X_test)
testing_time = time.time() - start_time


# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Training time: {training_time:.2f} seconds")
print(f"Testing time: {testing_time:.2f} seconds")
print(f"Decision Tree Accuracy: {accuracy:.2f}")