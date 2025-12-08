# Credit Card Fraud Detection - ML Fundamentals

A comprehensive machine learning project demonstrating classification, data preprocessing, model training, and evaluation on the credit card fraud detection dataset.

## 🎯 Problem Statement
Detect fraudulent credit card transactions from legitimate ones using supervised machine learning classification.

## 📊 Dataset
- **Source:** Credit Card Fraud Detection (Kaggle)
- **Samples:** ~284,000 transactions
- **Features:** 30 (PCA-transformed for privacy)
- **Class Distribution:** 99.8% legitimate, 0.2% fraudulent (imbalanced)
- **Download:** https://www.kaggle.com/mlg-ulb/creditcardfraud

## 🛠️ Tech Stack
- **Python 3.10+**
- **Pandas:** Data loading, cleaning, exploration
- **Scikit-learn:** Model training and evaluation
- **NumPy:** Numerical operations
- **Pickle:** Model serialization

## 📁 Project Structure
```
ml-crash-course/
├── 01_pandas_basics.py           # Load and explore data
├── 02_data_cleaning.py           # Clean missing values, duplicates
├── 03_train_test_split.py        # Split data properly
├── 04_eda.py                     # Exploratory Data Analysis
├── 05_train_model.py             # Train Random Forest model
├── 06_save_model.py              # Save/load model with pickle
├── 07_feature_importance.py      # Analyze feature importance
├── 08_compare_models.py          # Compare 3 algorithms
├── README.md                     # This file
└── creditcard.csv               # Dataset (download from Kaggle)
```

## 🚀 Quick Start

### Install dependencies
```bash
pip install pandas scikit-learn numpy matplotlib
```

### Download dataset
1. Go to https://www.kaggle.com/mlg-ulb/creditcardfraud
2. Download `creditcard.csv`
3. Place in project root folder

### Run scripts in order
```bash
python 01_pandas_basics.py
python 02_data_cleaning.py
python 03_train_test_split.py
python 04_eda.py
python 05_train_model.py        # Trains Random Forest
python 06_save_model.py         # Saves model to fraud_model.pkl
python 07_feature_importance.py # Shows which features matter
python 08_compare_models.py     # Compares Random Forest vs GradientBoosting vs AdaBoost
```

## 📊 Results

### Model Performance (Random Forest)
```
Accuracy:  0.9991 (99.91%)
Precision: 0.8876 (88.76%)  ← When we say FRAUD, we're right 89% of time
Recall:    0.6897 (68.97%)  ← We catch 69% of actual frauds
ROC-AUC:   0.9348
```

### Key Findings
1. **Class Imbalance:** Only 0.17% frauds → Accuracy alone is misleading
2. **Precision-Recall Trade-off:** High precision (avoid false alarms) vs High recall (catch frauds)
3. **Feature Importance:** Some features are much more predictive than others
4. **Model Comparison:** Gradient Boosting slightly outperforms Random Forest

## 🧠 What I Learned

### ML Concepts
- ✅ Classification problem (binary: fraud vs not fraud)
- ✅ Train/test split and why we need it
- ✅ Metrics: Accuracy, Precision, Recall, ROC-AUC
- ✅ Class imbalance and its impact

### Pandas Skills
- Loading data: `pd.read_csv()`
- Exploration: `.shape`, `.head()`, `.describe()`, `.isnull()`
- Cleaning: `.dropna()`, `.drop_duplicates()`
- Manipulation: `.drop()`, `.value_counts()`

### Scikit-learn Skills
- Train/test split: `train_test_split()`
- Model training: `RandomForestClassifier().fit()`
- Predictions: `model.predict()`
- Evaluation: `accuracy_score()`, `precision_score()`, `recall_score()`
- Model persistence: `pickle.dump()`, `pickle.load()`

## 🎯 Next Steps

This project is foundation for production MLOps system including:
- Feature engineering (velocity, behavioral features)
- Model versioning with MLflow
- Data versioning with DVC
- Real-time serving with FastAPI
- Deployment with Docker & Kubernetes
- Automated retraining with Airflow
- Monitoring with Evidently AI

## 📚 Resources Used
- Kaggle Credit Card Fraud Detection Dataset
- Scikit-learn documentation
- Pandas documentation

## 📧 Connect
- LinkedIn: https://www.linkedin.com/in/dinesh-k-narayanan/
- GitHub: https://github.com/DineshKNarayanankutty/

---

**Created:** December 2025  
**Status:** ✅ Completed (8 scripts, 3 models trained, metrics evaluated)
