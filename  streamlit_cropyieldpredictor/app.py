"""
Crop Yield Predictor - Streamlit App
--------------------------------------
Interactive web UI for the crop yield prediction model.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import os

st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌱",
    layout="centered"
)

DATA_PATH = os.path.join("data", "crop_yield_data.csv")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def train_model(df):
    le = LabelEncoder()
    df = df.copy()
    df["crop_encoded"] = le.fit_transform(df["crop"])

    features = ["crop_encoded", "rainfall_mm", "temperature_celsius",
                 "fertilizer_kg_per_ha", "soil_ph"]
    X = df[features]
    y = df["yield_tonnes_per_ha"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    return model, le, mae, r2


def main():
    st.title("🌱 Crop Yield Predictor")
    st.write(
        "Predict crop yield (tonnes per hectare) based on rainfall, "
        "temperature, fertilizer use, and soil pH."
    )

    df = load_data()
    model, encoder, mae, r2 = train_model(df)

    with st.sidebar:
        st.header("Model Info")
        st.metric("R² Score", f"{r2:.3f}")
        st.metric("Mean Absolute Error", f"{mae:.2f} t/ha")
        st.caption(f"Trained on {len(df)} records")

        st.divider()
        st.subheader("About")
        st.write(
            "This demo uses a Random Forest Regressor trained on a small "
            "sample dataset. Replace `data/crop_yield_data.csv` with your "
            "own data for real-world use."
        )

    st.subheader("Enter Field Conditions")

    crops = sorted(df["crop"].unique().tolist())

    col1, col2 = st.columns(2)
    with col1:
        crop = st.selectbox("Crop", crops)
        rainfall = st.slider("Rainfall (mm)", 100, 2000, 500, step=10)
        temperature = st.slider("Temperature (°C)", 5, 45, 25)

    with col2:
        fertilizer = st.slider("Fertilizer (kg/ha)", 0, 350, 130, step=5)
        soil_ph = st.slider("Soil pH", 4.0, 9.0, 6.5, step=0.1)

    if st.button("Predict Yield", type="primary", use_container_width=True):
        crop_encoded = encoder.transform([crop])[0]
        input_df = pd.DataFrame([{
            "crop_encoded": crop_encoded,
            "rainfall_mm": rainfall,
            "temperature_celsius": temperature,
            "fertilizer_kg_per_ha": fertilizer,
            "soil_ph": soil_ph
        }])

        prediction = model.predict(input_df)[0]

        st.success(f"### Predicted Yield: **{prediction:.2f} tonnes/ha**")

        # Show how prediction compares to dataset average for that crop
        crop_avg = df[df["crop"] == crop]["yield_tonnes_per_ha"].mean()
        diff = prediction - crop_avg
        delta_label = f"{diff:+.2f} t/ha vs avg"
        st.metric(
            label=f"Compared to average {crop} yield ({crop_avg:.2f} t/ha)",
            value=f"{prediction:.2f} t/ha",
            delta=delta_label
        )

    st.divider()

    with st.expander("📊 View Training Data"):
        st.dataframe(df, use_container_width=True)

    with st.expander("📈 Average Yield by Crop"):
        avg_yield = df.groupby("crop")["yield_tonnes_per_ha"].mean().sort_values(ascending=False)
        st.bar_chart(avg_yield)


if __name__ == "__main__":
    main()
