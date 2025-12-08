import pandas as pd
from sklearn.model_selection import train_test_split

file_path = 'creditcard.csv'
df = pd.read_csv(file_path)
df = df.dropna().drop_duplicates()

X = df.drop('Class', axis=1)
y = df['Class']

# SPLIT DATA: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing
    random_state=42,    # For reproducibility
    stratify=y          # Keep fraud/not-fraud ratio same in train and test
)

print(f"Total samples: {len(X)}")
print(f"Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test samples: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

print(f"\nTraining set - Fraud distribution:")
print(y_train.value_counts())
print(f"\nTest set - Fraud distribution:")
print(y_test.value_counts())
