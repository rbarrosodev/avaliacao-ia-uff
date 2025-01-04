import time
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Record the start time
start_time = datetime.now()
print("Start Time:", start_time)

param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'lbfgs', 'sgd'],
    'alpha': [0.0001],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'learning_rate_init': [0.0001]
}

# Load dataset
data = pd.read_csv("mobile-phone-data.csv")

# Separate features and target variable
X = data.drop("price_range", axis=1)  # Features
y = data["price_range"]  # Target variable

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(max_iter=1000, early_stopping=True, random_state=42)

# GridSearchCV for accuracy
grid_search_accuracy = GridSearchCV(estimator=mlp, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=2)

start_time_acc = time.time()
grid_search_accuracy.fit(X_train, y_train)
testing_time_acc = time.time() - start_time_acc

# Compare results
print(f"Testing time for Accuracy: {testing_time_acc:.2f} seconds")
print("Best Parameters for Accuracy:", grid_search_accuracy.best_params_)
print("Best Score for Accuracy:", grid_search_accuracy.best_score_)

time.sleep(10)

scoring_f1_methods = ['f1_micro', 'f1_macro', 'f1_weighted']

for scoring in scoring_f1_methods:
    grid_search_f1 = GridSearchCV(estimator=mlp, param_grid=param_grid, scoring=scoring,
                                  cv=5, n_jobs=-1, verbose=2)
    start_time_f1 = time.time()
    grid_search_f1.fit(X_train, y_train)
    testing_time_f1 = time.time() - start_time_f1
    # Melhor conjunto de parâmetros
    print(f"Testing time for {scoring}: {testing_time_f1:.2f} seconds")
    print(f"Best Parameters for {scoring}:", grid_search_f1.best_params_)
    print(f"Best Score for {scoring}:", grid_search_f1.best_score_)

    time.sleep(10)

# Record the end time
end_time = datetime.now()
print("End Time:", end_time)

# Calculate and print the duration
execution_time = end_time - start_time
print("Execution Time:", execution_time)