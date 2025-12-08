import pandas as pd

file_path = 'creditcard.csv'
df = pd.read_csv(file_path)

print("Original shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())

# Remove rows with missing values (this dataset has none, but good practice)
df = df.dropna()
print("After cleaning:", df.shape)

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"Duplicate rows: {duplicates}")

# Remove duplicates
df = df.drop_duplicates()
print("After removing duplicates:", df.shape)

# Separate features (X) and target (y)
X = df.drop('Class', axis=1)  # All columns except 'Class'
y = df['Class']  # Only 'Class' column

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)
print("Target classes:", y.unique())  # Should be [0, 1]
