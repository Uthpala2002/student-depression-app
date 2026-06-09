# 🧠 Student Depression Prediction App

This is a Machine Learning project that predicts student depression risk using XGBoost and Streamlit.

## 🚀 Features
- AI-powered prediction system
- Streamlit web app
- XGBoost model (~79% accuracy)
- Feature importance visualization

## 🎥 Demo Video
(https://drive.google.com/file/d/1-E_P2oXErlt8x2FyoXKeCSF2v0ZDnF9E/view?usp=sharing)

## 🛠 Tech Stack
- Python
- Scikit-learn
- XGBoost
- Streamlit
- Pandas

## ▶ How to run locally
```bash
streamlit run app.py

## 🚀 Version 2.0 Improvements

I recently refactored and upgraded this machine learning pipeline to improve robustness and predictive performance. Key enhancements include:

* **Feature Engineering & Data Preprocessing:** * Expanded the feature space from 6 to **14 relevant features**.
    * Integrated `StandardScaler` into the numerical pipeline to prevent scale bias.
    * Added `stratify=y` to the train-test split to ensure balanced class distributions across folds.
* **Model Optimization & Evaluation:**
    * Implemented `RandomizedSearchCV` to automate and optimize hyperparameter tuning.
    * Corrected the feature importance pipeline to utilize **XGBoost's intrinsic importance metrics** rather than Random Forest.
    * Refactored accuracy scoring to be computed **dynamically** rather than hardcoded.
* **Outputs & Usability:**
    * Added a **Confusion Matrix visualization** to better analyze true/false positives and negatives.
    * Upgraded the inference pipeline: the prediction function now returns the **class probability** alongside the final class label for better decision confidence.
