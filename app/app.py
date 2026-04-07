import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Define the paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

st.set_page_config(page_title="Diabetes Prediction", page_icon="🏥", layout="wide")

st.title("🏥 Diabetes Prediction App")
st.write("Enter the patient's medical details below to predict the likelihood of diabetes. The prediction engine runs on a robust High-Recall XGBoost architecture trained on over 100,000 records.")

# Load models
@st.cache_resource
def load_assets():
    models = {}
    try: models["XGBoost"] = joblib.load(os.path.join(MODEL_DIR, 'xgboost.pkl'))
    except Exception: pass

    return models

try:
    models = load_assets()
    models_loaded = len(models) > 0
except Exception as e:
    st.error(f"Error loading models. Please ensure they are trained and saved in the 'models' directory. Error: {e}")
    models_loaded = False

if models_loaded:
    selected_model = models["XGBoost"]

    st.header("Patient Data Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", options=["Female", "Male", "Other"])
        age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
        hypertension = st.selectbox("Hypertension", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        heart_disease = st.selectbox("Heart Disease", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        smoking_history = st.selectbox("Smoking History", options=["never", "No Info", "current", "former", "ever", "not current"])
        bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0, step=0.1)
        
    with col2:
        hba1c_level = st.number_input("HbA1c Level", min_value=3.0, max_value=20.0, value=5.5, step=0.1)
        blood_glucose_level = st.number_input("Blood Glucose Level", min_value=50, max_value=400, value=120, step=1)
        blood_pressure = st.number_input("Blood Pressure (Diastolic)", min_value=0, max_value=200, value=70, step=1)
        skin_thickness = st.number_input("Skin Thickness (Triceps)", min_value=0, max_value=100, value=20, step=1)
        insulin = st.number_input("Insulin Level", min_value=0, max_value=1000, value=79, step=1)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)

    if st.button("Predict", type="primary"):
        input_data = pd.DataFrame([[
            gender, age, hypertension, heart_disease, smoking_history, 
            bmi, hba1c_level, blood_glucose_level, blood_pressure, 
            skin_thickness, insulin, dpf
        ]], columns=[
            'gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 
            'bmi', 'HbA1c_level', 'blood_glucose_level', 'blood_pressure', 
            'skin_thickness', 'insulin', 'diabetes_pedigree_function'
        ])
        
        # Predict using the full pipeline
        prediction = selected_model.predict(input_data)[0]
        
        if hasattr(selected_model, "predict_proba"):
            probability = selected_model.predict_proba(input_data)[0][1]
            prob_text = f" (Probability: {probability:.2%})"
        else:
            prob_text = ""
            
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"🚨 The model predicts: **Diabetic**{prob_text}")
        else:
            st.success(f"✅ The model predicts: **Non-Diabetic**{prob_text}")
