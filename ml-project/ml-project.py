import pandas as pd
import numpy as np
import sklearn

print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print("Setup successful!")

from sklearn.datasets import fetch_california_housing

# Load dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

print(f"Dataset shape: {df.shape}")
print(f"Features: {list(housing.feature_names)}")
print("\nFirst few rows:")
print(df.head())

print(df.describe())

print(f"\nMissing values: {df.isnull().sum().sum()}")

from sklearn.model_selection import train_test_split

# Separate features (X) from target (y)
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Split into 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")