import time
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Record the start time
start_time = datetime.now()
print("Start Time:", start_time)

param_grid = {
    'n_estimators': [200, 300],          # Number of trees in the forest
    'max_depth': [2, 5, 10, 15, 20, None],           # Maximum depth of the tree
    'min_samples_split': [2, 5, 10, 20, 50],
    'min_samples_leaf': [1, 2, 5, 10, 20],             # Minimum number of samples in a leaf node
    'bootstrap': [True],                    # Whether bootstrap samples are used when building trees                 # Whether bootstrap samples are used when building trees
    'oob_score': [True, False],
    'max_features': ['sqrt', 'log2', None],
    # Whether to use out-of-bag samples to estimate the generalization error
    'class_weight': ['balanced', None]
}

# Load dataset
data = pd.read_csv("mobile-phone-data.csv")

# Separate features and target variable
X = data.drop("price_range", axis=1)  # Features
y = data["price_range"]  # Target variable

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train Random Forest model
rf = RandomForestClassifier(random_state=42)

grid_search_accuracy = GridSearchCV(estimator=rf, param_grid=param_grid,
                                    scoring='accuracy', cv=5, n_jobs=-1, verbose=2)

# Initialize and train Decision Tree model

start_time_acc = time.time()
grid_search_accuracy.fit(X_train, y_train)
testing_time_acc = time.time() - start_time_acc
print(f"Testing time for Accuracy: {testing_time_acc:.2f} seconds")
print("Best Parameters for Accuracy:", grid_search_accuracy.best_params_)
print("Best Score for Accuracy:", grid_search_accuracy.best_score_)

scoring_f1_methods = ['f1_micro', 'f1_macro', 'f1_weighted']

for scoring in scoring_f1_methods:
    grid_search_f1 = GridSearchCV(estimator=rf, param_grid=param_grid, scoring=scoring,
                                  cv=5, n_jobs=-1, verbose=2)
    start_time_f1 = time.time()
    grid_search_f1.fit(X_train, y_train)
    testing_time_f1 = time.time() - start_time_f1
    # Melhor conjunto de parâmetros
    print(f"Testing time for {scoring}: {testing_time_f1:.2f} seconds")
    print(f"Best Parameters for {scoring}:", grid_search_f1.best_params_)
    print(f"Best Score for {scoring}:", grid_search_f1.best_score_)

# Record the end time
end_time = datetime.now()
print("End Time:", end_time)

# Calculate and print the duration
execution_time = end_time - start_time
print("Execution Time:", execution_time)