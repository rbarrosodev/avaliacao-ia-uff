import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

# Load dataset
data = pd.read_csv("mobile-phone-data.csv")

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


# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Training time: {training_time:.2f} seconds")
print(f"Testing time: {testing_time:.2f} seconds")
print(f"Decision Tree Accuracy: {accuracy:.2f}")

scoring_f1_methods = ['micro']

for scoring in scoring_f1_methods:
    dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=10, max_features=None,
                                  min_samples_leaf=5, min_samples_split=2, random_state=42)

    start_time_f1_training = time.time()
    dt_model.fit(X_train, y_train)
    training_time_f1 = time.time() - start_time_f1_training

    start_time_f1_testing = time.time()
    y_pred = dt_model.predict(X_test)
    testing_time_f1 = time.time() - start_time_f1_testing

    f1 = f1_score(y_test, y_pred, average=scoring)
    # Melhor conjunto de parâmetros
    print(f"Training time for F1 {scoring}: {training_time_f1:.2f} seconds")
    print(f"Testing time for F1 {scoring}: {testing_time_f1:.2f} seconds")
    print(f"Decision Tree Score for F1 {scoring}: {f1:.2f}")

    precision = precision_score(y_test, y_pred, average=scoring)  # or 'binary' for binary classification
    recall = recall_score(y_test, y_pred, average=scoring)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Class distribution:", np.bincount(y_test))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)