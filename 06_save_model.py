import pickle
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

# SAVE MODEL
with open('fraud_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ Model saved to fraud_model.pkl")

# LOAD MODEL (to verify)
with open('fraud_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
print("✅ Model loaded from fraud_model.pkl")

# Test loaded model
score = loaded_model.score(X_test, y_test)
print(f"✅ Loaded model accuracy: {score:.4f}")
