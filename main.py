import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("weather-data.csv")

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
y = label_encoder.fit_transform(data["RainTomorrow"])               # Target variable

# Number of rows in X
num_rows = len(X)
print(f"Number of rows in X: {num_rows}")

# Number of rows with null values in X
num_null_rows = X.isnull().any(axis=1).sum()
print(f"Number of rows with null values in X: {num_null_rows}")

# Number of rows in X
num_rows = len(y)
print(f"Number of rows in y: {num_rows}")

# Number of rows with null values in X
num_null_rows = y.isnull().any(axis=1).sum()
print(f"Number of rows with null values in y: {num_null_rows}")