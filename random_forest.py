import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("mobile-phone-data.csv")

# Separate features and target variable
X = data.drop("price_range", axis=1)  # Features
y = data["price_range"]  # Target variable

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train Random Forest model
rf_model = RandomForestClassifier(bootstrap=True, max_depth=None,
                                  min_samples_leaf=5, min_samples_split=10, oob_score=True,
                                  n_estimators=200, random_state=42)
start_time = time.time()
rf_model.fit(X_train, y_train)
training_time = time.time() - start_time


# Make predictions
start_time = time.time()
y_pred = rf_model.predict(X_test)
testing_time = time.time() - start_time


# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Training time: {training_time:.2f} seconds")
print(f"Testing time: {testing_time:.2f} seconds")
print(f"Random Forest Accuracy: {accuracy:.2f}")