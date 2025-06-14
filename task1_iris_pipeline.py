# Task 1: Simple Data Pipeline using Iris Dataset

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

# Step 1: Extract - Load the dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Simulate missing data for the sake of preprocessing
df.loc[2, 'sepal length (cm)'] = np.nan
df.loc[5, 'species'] = np.nan

print("🔹 Raw data sample:")
print(df.head())

# Step 2: Transform

# Handle missing numerical values with mean
num_imputer = SimpleImputer(strategy='mean')
df[iris.feature_names] = num_imputer.fit_transform(df[iris.feature_names])

# Handle missing categorical values with mode
df['species'].fillna(df['species'].mode()[0], inplace=True)

# Encode categorical column (species)
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])  # setosa=0, versicolor=1, virginica=2

# Scale numeric columns between 0 and 1
scaler = MinMaxScaler()
df[iris.feature_names] = scaler.fit_transform(df[iris.feature_names])

print("\n🔹 Cleaned & Transformed data sample:")
print(df.head())

# Step 3: Load - Save to CSV
df.to_csv("cleaned_iris.csv", index=False)
print("\n✅ Cleaned data saved as 'cleaned_iris.csv'")
