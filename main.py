import time
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("mobile-phone-data.csv")

print(data.isnull().sum())  # Check for missing values in each column

#print(data.dtypes)