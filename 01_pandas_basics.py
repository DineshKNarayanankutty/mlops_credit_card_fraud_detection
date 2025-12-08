import pandas as pd

# Load fraud dataset
file_path = 'creditcard.csv'
df = pd.read_csv(file_path)

# BASIC OPERATIONS
print("Shape:", df.shape)  # How many rows and columns?
print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nBasic stats:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

print("\nClass distribution (fraud vs not fraud):")
print(df['Class'].value_counts())
# Class = 0 means not fraud, Class = 1 means fraud
