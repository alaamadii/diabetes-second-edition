import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

st.set_page_config(page_title="Diabetes Classification (XAI)", page_icon="🏥", layout="wide")
st.title("🏥 Diabetes Classification Demo")
st.write(
    "Educational machine-learning demo using an XGBoost pipeline. "
    "The output is a model prediction and score, not a diagnosis or clinical recommendation."
)


@st.cache_resource
def load_assets():
    model_path = os.path.join(MODEL_DIR, "xgboost.pkl")
    return {"XGBoost": joblib.load(model_path)}


try:
    models = load_assets()
except Exception as exc:
    st.error(
        "Could not load the trained model from the models directory. "
        f"Details: {exc}"
    )
    models = {}

if models:
    selected_model = models["XGBoost"]

    st.header("Input Features")
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", options=["Female", "Male", "Other"])
        age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
        hypertension = st.selectbox(
            "Hypertension",
            options=[0, 1],
            format_func=lambda value: "Yes" if value == 1 else "No",
        )
        heart_disease = st.selectbox(
            "Heart Disease",
            options=[0, 1],
            format_func=lambda value: "Yes" if value == 1 else "No",
        )
        smoking_history = st.selectbox(
            "Smoking History",
            options=["never", "No Info", "current", "former", "ever", "not current"],
        )
        bmi = st.number_input(
            "BMI", min_value=10.0, max_value=100.0, value=25.0, step=0.1
        )

    with col2:
        hba1c_level = st.number_input(
            "HbA1c Level", min_value=3.0, max_value=20.0, value=5.5, step=0.1
        )
        blood_glucose_level = st.number_input(
            "Blood Glucose Level", min_value=50, max_value=400, value=120, step=1
        )
        blood_pressure = st.number_input(
            "Blood Pressure (Diastolic)",
            min_value=0,
            max_value=200,
            value=70,
            step=1,
        )
        skin_thickness = st.number_input(
            "Skin Thickness (Triceps)",
            min_value=0,
            max_value=100,
            value=20,
            step=1,
        )
        insulin = st.number_input(
            "Insulin Level", min_value=0, max_value=1000, value=79, step=1
        )
        dpf = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.5,
            step=0.01,
        )

    if st.button("Run model & explain", type="primary"):
        input_data = pd.DataFrame(
            [
                [
                    gender,
                    age,
                    hypertension,
                    heart_disease,
                    smoking_history,
                    bmi,
                    hba1c_level,
                    blood_glucose_level,
                    blood_pressure,
                    skin_thickness,
                    insulin,
                    dpf,
                ]
            ],
            columns=[
                "gender",
                "age",
                "hypertension",
                "heart_disease",
                "smoking_history",
                "bmi",
                "HbA1c_level",
                "blood_glucose_level",
                "blood_pressure",
                "skin_thickness",
                "insulin",
                "diabetes_pedigree_function",
            ],
        )

        prediction = selected_model.predict(input_data)[0]
        score_text = ""
        if hasattr(selected_model, "predict_proba"):
            model_score = selected_model.predict_proba(input_data)[0][1]
            score_text = f" Model score: {model_score:.2%}."

        st.subheader("Model Output")
        if prediction == 1:
            st.warning(f"Model class: positive.{score_text}")
        else:
            st.info(f"Model class: negative.{score_text}")

        st.caption(
            "The displayed score is the classifier output from this experimental pipeline. "
            "The repository does not establish that it is a calibrated clinical probability."
        )

        st.write("---")
        st.subheader("🧠 SHAP explanation")
        st.write(
            "The waterfall plot shows how transformed features contributed to this model output. "
            "SHAP attribution does not establish causality or clinical validity."
        )

        try:
            preprocessor = selected_model.named_steps["preprocessor"]
            classifier = selected_model.named_steps["classifier"]
            input_processed = preprocessor.transform(input_data)

            try:
                raw_names = preprocessor.get_feature_names_out()
                feature_names = [
                    name.split("__")[-1].replace("_", " ").title()
                    for name in raw_names
                ]
            except Exception:
                feature_names = [
                    f"Feature {index}" for index in range(input_processed.shape[1])
                ]

            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer(input_processed)
            shap_values.feature_names = feature_names

            fig = plt.figure(figsize=(10, 6))
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as exc:
            st.error(f"Could not generate the SHAP plot. Details: {exc}")
