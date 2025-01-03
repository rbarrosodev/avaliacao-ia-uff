import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("weather-data.csv")

# Set display option to show all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)  # Optional: Show all rows (be cautious with large DataFrames)

# Limpeza da base
data['Date'] = pd.to_datetime(data['Date'])

# Extract features like year, month, day, etc.
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

# Drop the original 'Date' column
data = data.drop('Date', axis=1)

data = pd.get_dummies(data, columns=['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday'])

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Separate features and target variable
X = data.drop("RainTomorrow", axis=1)  # Features
y = data["RainTomorrow"]  # Target variable
missing_y_indices = y.isnull()

X_clean = X[~missing_y_indices]  # Keep rows where y is not null
y_clean = y[~missing_y_indices]  # Keep rows where y is not null

y_clean = label_encoder.fit_transform(y_clean)


# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.3, random_state=42)

# Initialize and train Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
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