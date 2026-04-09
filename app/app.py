import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt

# Define the paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

st.set_page_config(page_title="Diabetes Prediction (XAI)", page_icon="🏥", layout="wide")

st.title("🏥 Diabetes Prediction App")
st.write("Enter the patient's medical details below to predict the likelihood of diabetes. The prediction engine runs on a robust High-Recall XGBoost architecture trained on over 100,000 records. Explainable AI (SHAP) generates visual summaries of *why* the model made its decision.")

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

    if st.button("Predict & Explain", type="primary"):
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
            st.error(f"🚨 The model flags the patient as: **High Risk (Diabetic)**{prob_text}")
        else:
            st.success(f"✅ The model clears the patient as: **Low Risk (Non-Diabetic)**{prob_text}")
            
        # ====================
        # EXPLAINABLE AI (XAI)
        # ====================
        st.write("---")
        st.subheader("🧠 AI Explanation (SHAP)")
        st.write("This chart explains exactly which patient factors drove the clinical prediction.")
        
        try:
            # 1. Break the pipeline apart to access the raw Random Forest / XGBoost model internally
            preprocessor = selected_model.named_steps['preprocessor']
            classifier = selected_model.named_steps['classifier']
            
            # 2. Process the single patient's data through the imputation and scaling rules
            input_processed = preprocessor.transform(input_data)
            
            # 3. Pull human readable feature mapping from the preprocessor encodings
            feature_names = []
            try:
                # Get dynamic encoded names (e.g., 'cat__gender_Female')
                raw_names = preprocessor.get_feature_names_out()
                # Clean up prefixes so visual presentation is clean
                for name in raw_names:
                    clean_name = name.split('__')[-1].replace('_', ' ').title()
                    feature_names.append(clean_name)
            except Exception:
                feature_names = [f"Feature {i}" for i in range(input_processed.shape[1])]
            
            # 4. Generate the SHAP Explainer
            # TreeExplainer is lightning fast for XGBoost models
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer(input_processed)
            
            # Reassign clean Feature Names rather than arrays indices
            shap_values.feature_names = feature_names
            
            # 5. Render SHAP Waterfall inside Streamlit
            # We must map matplotlib's global pyplot instance safely inside Streamlit
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)
            plt.clf() # Free up memory
            
        except Exception as e:
            st.error(f"Failed to generate Explainable AI summary graph for this specific architecture. Details: {e}")
