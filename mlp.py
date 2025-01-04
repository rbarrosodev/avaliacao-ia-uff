import time
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score
import numpy as np
from sklearn.metrics import confusion_matrix


# Load dataset
data = pd.read_csv("mobile-phone-data.csv")

# Separate features and target variable
X = data.drop("price_range", axis=1)  # Features
y = data["price_range"]  # Target variable

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp_model = MLPClassifier(activation='tanh', alpha=0.0001, learning_rate='constant',
                          hidden_layer_sizes=(100,100), max_iter=1000, early_stopping=True,
                          random_state=42, learning_rate_init=0.0001, solver='lbfgs')

start_time = time.time()
mlp_model.fit(X_train_scaled, y_train)
training_time = time.time() - start_time


# Make predictions
start_time = time.time()
y_pred = mlp_model.predict(X_test_scaled)
testing_time = time.time() - start_time


# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Training time: {training_time:.2f} seconds")
print(f"Testing time: {testing_time:.2f} seconds")
print(f"MLP Accuracy: {accuracy:.2f}")

scoring_f1_methods = ['micro']

for scoring in scoring_f1_methods:
    mlp_model = MLPClassifier(activation='tanh', alpha=0.0001, learning_rate='constant',
                              hidden_layer_sizes=(100, 100), max_iter=1000, early_stopping=True,
                              random_state=42, learning_rate_init=0.0001, solver='lbfgs')
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    start_time_f1_training = time.time()
    mlp_model.fit(X_train_scaled, y_train)
    training_time_f1 = time.time() - start_time_f1_training

    start_time_f1_testing = time.time()
    y_pred = mlp_model.predict(X_test_scaled)
    testing_time_f1 = time.time() - start_time_f1_testing

    f1 = f1_score(y_test, y_pred, average=scoring)
    # Melhor conjunto de parâmetros
    print(f"Training time for F1 {scoring}: {training_time_f1:.2f} seconds")
    print(f"Testing time for F1 {scoring}: {testing_time_f1:.2f} seconds")
    print(f"MLP Score for F1 {scoring}: {f1:.2f}")

    precision = precision_score(y_test, y_pred, average=scoring)  # or 'binary' for binary classification
    recall = recall_score(y_test, y_pred, average=scoring)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Class distribution:", np.bincount(y_test))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
