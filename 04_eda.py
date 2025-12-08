import pandas as pd
import matplotlib.pyplot as plt

file_path = 'creditcard.csv'
df = pd.read_csv(file_path)

# Basic stats
print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

# Fraud distribution
print("\nFraud distribution:")
fraud_counts = df['Class'].value_counts()
print(fraud_counts)
print(f"\nFraud percentage: {fraud_counts[1]/len(df)*100:.2f}%")

# Statistical summary
print("\nStatistical summary:")
print(df.describe())

# Correlation with fraud (optional, for understanding)
print("\nTop 10 features most correlated with fraud:")
correlation = df.corr()['Class'].sort_values(ascending=False)
print(correlation.head(10))
