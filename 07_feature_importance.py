import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load and prepare data
file_path = 'creditcard.csv'
df = pd.read_csv(file_path)
df = df.dropna().drop_duplicates()

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# GET FEATURE IMPORTANCE
importances = model.feature_importances_
feature_names = X.columns

# Sort by importance
indices = sorted(range(len(importances)),
                 key=lambda i: importances[i], reverse=True)

print("Top 10 Most Important Features:")
print("="*50)
for i, idx in enumerate(indices[:10]):
    print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

print("\n" + "="*50)
print("INSIGHT:")
print("="*50)
print("These features are most useful for detecting fraud.")
print("In production, we'd focus on engineering features similar to these.")
