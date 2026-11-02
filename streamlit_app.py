"""Streamlit interface for the saved Starbucks beverage classifier."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent


def artifact_path(filename: str) -> Path:
    """Find artifacts whether GitHub upload placed them at root or in models/."""
    for folder in (ROOT / "models", ROOT):
        candidate = folder / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find {filename} in the repository root or models/ folder")


@st.cache_resource
def load_artifacts():
    model = joblib.load(artifact_path("starbucks_perfect_model.pkl"))
    scaler = joblib.load(artifact_path("starbucks_perfect_scaler.pkl"))
    target_encoder = joblib.load(artifact_path("starbucks_perfect_target_encoder.pkl"))
    return model, scaler, target_encoder


st.set_page_config(page_title="Beverage Insights", page_icon="☕", layout="centered")
st.title("☕ BEVERAGE INSIGHTS")
st.subheader("Starbucks Beverage Classifier")
st.write("Enter a drink's encoded type and nutrition values to see the model's predicted category.")

st.info(
    "This is a student ML demonstration, not medical or nutrition advice. "
    "The Beverage and Beverage prep encoders were not saved, so use only the "
    "correct integer codes from the original training data."
)

try:
    model, scaler, target_encoder = load_artifacts()
except Exception as exc:
    st.error(f"Could not load the model files from the models folder: {exc}")
    st.stop()

features = list(scaler.feature_names_in_)

with st.form("prediction_form"):
    st.markdown("#### Drink type")
    left, right = st.columns(2)
    with left:
        beverage_code = st.number_input(
            "Beverage code",
            min_value=0,
            value=0,
            step=1,
            help="Integer label from the Beverage encoder used during training.",
        )
    with right:
        preparation_code = st.number_input(
            "Beverage preparation code",
            min_value=0,
            value=0,
            step=1,
            help="Integer label from the Beverage_prep encoder used during training.",
        )

    st.markdown("#### Nutrition values")
    entries = [
        ("Calories", "calories"),
        ("Total Fat (g)", "total_fat"),
        ("Trans Fat (g)", "trans_fat"),
        ("Saturated Fat (g)", "saturated_fat"),
        ("Sodium (mg)", "sodium"),
        ("Total Carbohydrates (g)", "total_carbohydrates"),
        ("Cholesterol (mg)", "cholesterol"),
        ("Dietary Fibre (g)", "dietary_fibre"),
        ("Sugars (g)", "sugars"),
        ("Protein (g)", "protein"),
        ("Vitamin A (% DV)", "vitamin_a_dv"),
        ("Vitamin C (% DV)", "vitamin_c_dv"),
        ("Calcium (% DV)", "calcium_dv"),
        ("Iron (% DV)", "iron_dv"),
        ("Caffeine (mg)", "caffeine"),
    ]
    values = {"beverage_code": float(beverage_code), "preparation_code": float(preparation_code)}
    columns = st.columns(2)
    for index, (label, key) in enumerate(entries):
        with columns[index % 2]:
            values[key] = st.number_input(label, min_value=0.0, value=0.0, step=1.0, key=key)

    submitted = st.form_submit_button("Classify beverage", type="primary", use_container_width=True)

if submitted:
    ordered_values = [
        values["beverage_code"],
        values["preparation_code"],
        values["calories"],
        values["total_fat"],
        values["trans_fat"],
        values["saturated_fat"],
        values["sodium"],
        values["total_carbohydrates"],
        values["cholesterol"],
        values["dietary_fibre"],
        values["sugars"],
        values["protein"],
        values["vitamin_a_dv"],
        values["vitamin_c_dv"],
        values["calcium_dv"],
        values["iron_dv"],
        values["caffeine"],
    ]
    try:
        input_frame = pd.DataFrame([ordered_values], columns=features)
        transformed = scaler.transform(input_frame)
        encoded_prediction = model.predict(transformed).astype(int)
        prediction = target_encoder.inverse_transform(encoded_prediction)[0]

        st.success(f"Predicted category: **{prediction}**")
        if hasattr(model, "predict_proba"):
            confidence = float(model.predict_proba(transformed).max())
            st.caption(f"Model confidence: {confidence:.1%} (not a guarantee of correctness).")
    except Exception as exc:
        st.error(f"Could not classify these inputs: {exc}")
