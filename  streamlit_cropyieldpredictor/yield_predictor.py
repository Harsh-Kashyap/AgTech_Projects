"""
Crop Yield Predictor
---------------------
A simple machine learning project that predicts crop yield (tonnes/hectare)
based on environmental and farming inputs:
- Crop type
- Rainfall (mm)
- Temperature (Celsius)
- Fertilizer used (kg/ha)
- Soil pH

Uses a Random Forest Regressor from scikit-learn.

Usage:
    python yield_predictor.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os


DATA_PATH = os.path.join("data", "crop_yield_data.csv")
MODEL_PATH = os.path.join("model", "yield_model.pkl")
ENCODER_PATH = os.path.join("model", "crop_encoder.pkl")


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df


def preprocess(df):
    le = LabelEncoder()
    df["crop_encoded"] = le.fit_transform(df["crop"])
    features = ["crop_encoded", "rainfall_mm", "temperature_celsius",
                 "fertilizer_kg_per_ha", "soil_ph"]
    X = df[features]
    y = df["yield_tonnes_per_ha"]
    return X, y, le


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"Model trained.")
    print(f"  Mean Absolute Error: {mae:.3f} tonnes/ha")
    print(f"  R^2 Score: {r2:.3f}")

    return model


def save_model(model, encoder):
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    print(f"Model saved to {MODEL_PATH}")


def predict_yield(model, encoder, crop, rainfall, temperature, fertilizer, soil_ph):
    crop_encoded = encoder.transform([crop])[0]
    input_df = pd.DataFrame([{
        "crop_encoded": crop_encoded,
        "rainfall_mm": rainfall,
        "temperature_celsius": temperature,
        "fertilizer_kg_per_ha": fertilizer,
        "soil_ph": soil_ph
    }])
    prediction = model.predict(input_df)[0]
    return prediction


def main():
    print("=" * 50)
    print("Crop Yield Predictor")
    print("=" * 50)

    df = load_data()
    print(f"\nLoaded {len(df)} records covering crops: {df['crop'].unique().tolist()}\n")

    X, y, encoder = preprocess(df)
    model = train_model(X, y)
    save_model(model, encoder)

    # Example prediction
    print("\n--- Example Prediction ---")
    example = {
        "crop": "Wheat",
        "rainfall": 520,
        "temperature": 22,
        "fertilizer": 135,
        "soil_ph": 6.5
    }
    result = predict_yield(model, encoder, **example)
    print(f"Inputs: {example}")
    print(f"Predicted yield: {result:.2f} tonnes/ha")

    # Interactive prediction
    print("\n--- Try Your Own Prediction ---")
    try:
        crop = input(f"Enter crop {df['crop'].unique().tolist()}: ").strip()
        if crop not in df["crop"].unique():
            print("Unknown crop, skipping interactive prediction.")
            return
        rainfall = float(input("Rainfall (mm): "))
        temperature = float(input("Temperature (Celsius): "))
        fertilizer = float(input("Fertilizer (kg/ha): "))
        soil_ph = float(input("Soil pH: "))

        result = predict_yield(model, encoder, crop, rainfall, temperature, fertilizer, soil_ph)
        print(f"\nPredicted yield for {crop}: {result:.2f} tonnes/ha")
    except (EOFError, KeyboardInterrupt):
        print("\nSkipping interactive prediction.")


if __name__ == "__main__":
    main()
