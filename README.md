# 💳 Credit Card Fraud Detection (Machine Learning)

## 📌 Overview

This project focuses on detecting fraudulent credit card transactions using Machine Learning techniques.
The dataset is highly imbalanced, so techniques like SMOTE are used to improve fraud detection performance.

---

## 🚀 Features

* Data Cleaning & Preprocessing
* Handling Imbalanced Data using SMOTE
* Feature Encoding (Categorical to Numerical)
* Model Training (Random Forest)
* Model Evaluation using Precision, Recall, F1-score

---

## 📊 Dataset Info

* Total Transactions: ~2500
* Fraud Cases: 91
* Non-Fraud Cases: 2409
* Target Column: `is_fraud`

---

## ⚙️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Imbalanced-learn (SMOTE)

---

## 🧠 Machine Learning Workflow

1. Data Loading
2. Data Cleaning
3. Feature Engineering
4. Train-Test Split
5. Handling Imbalance (SMOTE)
6. Model Training
7. Evaluation

---

## 📈 Model Performance

* Focused on improving **Recall for fraud detection**
* Evaluated using:

  * Precision
  * Recall
  * F1-score

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python main.py
```

---

## 💡 Key Insight

Handling class imbalance is crucial in fraud detection problems.
Using SMOTE significantly improves the model’s ability to detect fraudulent transactions.

---

## 📌 Future Improvements

* Hyperparameter Tuning
* XGBoost / LightGBM
* Deployment using Streamlit

---

## 👨‍💻 Author

Mohit (Data Science Enthusiast)
