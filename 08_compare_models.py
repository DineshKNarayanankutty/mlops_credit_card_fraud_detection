import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import recall_score, precision_score

# Load and prepare data
file_path = 'creditcard.csv'
df = pd.read_csv(file_path)
df = df.dropna().drop_duplicates()

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Define models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42)
}

print("="*60)
print("MODEL COMPARISON")
print("="*60)

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")

print("\n" + "="*60)
print("CONCLUSION:")
print("="*60)
print("Different models give different results.")
print("We pick the one with best RECALL (catch most frauds).")
print("This is what we'll do in MLflow (track all experiments).")
