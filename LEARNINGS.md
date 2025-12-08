# 🚀 Machine Learning Crash Course — What I Learned

This is a quick summary of the core concepts I picked up while working through an ML crash course, especially focused on fraud-detection tasks.

---

## 📊 Working with Pandas

Pandas makes it easy to load, inspect, and clean datasets.

* Load data: `pd.read_csv(url)`
* Explore your dataset: `.shape`, `.head()`, `.describe()`
* Clean it up:

  * Remove missing values: `.dropna()`
  * Remove duplicates: `.drop_duplicates()`
  * Drop a column: `.drop('column', axis=1)`
* Prepare features and target:

  ```python
  X = df.drop('Class', axis=1)
  y = df['Class']
  ```

---

## ✂️ Train/Test Split — Why It Matters

We always keep some data *unseen* so we can measure real-world performance later.

* Use `train_test_split(X, y, test_size=0.2, stratify=y)`
* This gives:

  * **80%** for training
  * **20%** for testing
* `stratify=y` keeps the class distribution consistent, which is crucial for imbalanced datasets.

---

## 🤖 Building Models with Scikit-learn

The basic workflow:

1. Create a model

   ```python
   RandomForestClassifier(n_estimators=100)
   ```
2. Train it:

   ```python
   model.fit(X_train, y_train)
   ```
3. Make predictions:

   ```python
   model.predict(X_test)
   ```
4. Evaluate using:

   * `accuracy_score`
   * `precision_score`
   * `recall_score`

---

## 📈 Understanding Key Metrics

Different metrics tell different stories:

* **Accuracy** — % of correct predictions
  *Not useful when classes are imbalanced.*

* **Precision** — Of all predicted frauds, how many are actually fraud?
  *Helps reduce false alarms.*

* **Recall** — Of all actual frauds, how many did we catch?
  *Super important — missing fraud is costly.*

* **ROC-AUC** — Measures overall model quality (higher is better).

---

## 🔍 Insights for Fraud Detection

Fraud detection is especially tricky because:

* The dataset is **extremely imbalanced** (fraud ≈ 0.17%)
* A model can score **99% accuracy** by simply predicting “not fraud” every time
* Therefore, **accuracy is not the goal**

What matters:

* **High Recall** → Catch as many fraud cases as possible
* **Reasonable Precision** → Avoid overwhelming analysts with false alarms
* Good models often aim for **Recall > 80%**
