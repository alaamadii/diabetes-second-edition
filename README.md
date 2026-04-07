# 🏥 Diabetes Classification Project (XGBoost High-Recall)

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-scikit--learn%20%7C%20xgboost-orange)
![Web App](https://img.shields.io/badge/Web%20App-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📖 Overview

The **Diabetes Classification Project** is a powerful clinical Machine Learning solution designed to predict the likelihood of diabetes in patients. 

This repository boasts a **state-of-the-art XGBoost** inference engine trained on a massively expanded dataset of **100,768 clinical records**. Instead of prioritizing crude mathematical accuracy, the algorithm has been explicitly configured as a **High-Recall triage tool**, capable of catching 88% of actual diabetic patients to prevent dangerous false-negative clinical discharges.

The model comes wrapped in a responsive **Streamlit** user interface allowing clinicians to instantly flag high-risk metabolic profiles based on 12 distinct health features.

## ✨ Features

- **Large-Scale Medical Dataset:** Merged and normalized a massive 100k+ dataset incorporating features like `HbA1c`, `Smoking History`, and `Heart Disease`.
- **Advanced Imputation & Pipelines:** Seamless handling of missing clinical data points using standard ML pipelines.
- **High-Recall XGBoost Architecture:** Applied a severe positive-class scaling weight (`10.49x`) directly inside the engine to ensure maximum sensitivity (88% Recall).
- **Interactive Web Application:** A Streamlit-based UI that processes patient input arrays in real-time.

## 🛠️ Technology Stack

- **Data Wrangling:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn`, `xgboost`
- **Model Persistence:** `joblib`
- **Web App Framework:** `streamlit`

## 📂 Directory Structure

```text
├── app/
│   └── app.py                     # Streamlit web application source code
├── data/
│   └── merged_diabetes.csv        # Massive 100k+ clinical dataset
├── models/
│   └── xgboost.pkl                # Serialized full XGBoost preprocessing & inference pipeline
├── notebooks/
│   ├── clean_data.py              # Data cleaning scripts
│   ├── eda_script.py              # Exploratory Data Analysis
│   ├── data_merging.py            # Dataset merge and standardization script
│   └── pipeline_and_evaluation.py # Model evaluation & pipeline tests
├── train_models.py                # Main script to trigger XGBoost training & metric validation
├── .gitignore                     # Git ignored files
└── README.md                      # Project documentation
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/diabetes-classification-xgboost.git
cd diabetes-classification-xgboost
```

### 2. Install Dependencies

Ensure you have Python 3.8+ installed. Install the required libraries:

```bash
pip install pandas numpy scikit-learn xgboost joblib streamlit
```

### 3. Train the Model

To regenerate the `xgboost.pkl` file and view the live recall metrics, run the training pipeline:

```bash
python train_models.py
```

### 4. Run the Clinical Application

Launch the Streamlit app into your local browser to infer predictions visually:

```bash
streamlit run app/app.py
```

## 📈 High-Recall Strategy & Results

In medical screening, missing a diabetic patient (False Negative) is inherently far more dangerous than requesting a healthy patient to undergo a follow-up test (False Positive).

To combat the heavy dataset imbalance (`~91,000` healthy vs `~9,000` diabetic), we utilized XGBoost's `scale_pos_weight` parameter to strictly enforce minority class prioritization.

| Model | Overall Accuracy | Diabetic Recall (Sensitivity) | Healthy Recall (Specificity) |
|---|---:|---:|---:|
| **XGBoost Classifier** | **91.90%** | **88%** | 92% |

### ✅ Clinical Verdict
The inference engine successfully identified **88%** of actual sick patients, establishing it as an incredibly sound preliminary triage tool for hospitals or clinics.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! 



