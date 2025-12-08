import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Load and prepare data
file_path = 'creditcard.csv'
df = pd.read_csv(file_path)
df = df.dropna().drop_duplicates()

X = df.drop('Class', axis=1)
y = df['Class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# CREATE AND TRAIN MODEL
print("Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,   # 100 decision trees
    random_state=42,
    n_jobs=-1           # Use all CPU cores
)
model.fit(X_train, y_train)
print("✅ Model trained!")

# MAKE PREDICTIONS
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of fraud

print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\n" + "="*50)
print("INTERPRETATION")
print("="*50)
print(f"✓ Accuracy: {accuracy*100:.2f}% of transactions classified correctly")
print(
    f"✓ Precision: When we say FRAUD, we're right {precision*100:.2f}% of the time")
print(f"✓ Recall: We catch {recall*100:.2f}% of actual frauds")
print(f"✓ ROC-AUC: Overall model quality score (1.0 = perfect)")

if recall < 0.8:
    print("\n⚠️  WARNING: Recall is low. We're missing too many frauds!")
    print("    Need to adjust model or features.")
