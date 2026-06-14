# 🌱 Crop Yield Predictor

A simple machine learning project that predicts crop yield (tonnes/hectare) based on:

- Crop type (Wheat, Rice, Maize, Cotton, Soybean)
- Rainfall (mm)
- Temperature (°C)
- Fertilizer used (kg/ha)
- Soil pH

Built with **pandas** and **scikit-learn** (Random Forest Regressor).

## Project Structure

```
crop-yield-predictor/
├── data/
│   └── crop_yield_data.csv   # sample training data
├── model/                     # trained model saved here (created on run)
├── yield_predictor.py         # main script
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python yield_predictor.py
```

This will:
1. Load the sample dataset
2. Train a Random Forest model
3. Print evaluation metrics (MAE, R²)
4. Save the trained model to `model/`
5. Show an example prediction
6. Prompt you to enter your own values for a custom prediction

## Example Output

```
Model trained.
  Mean Absolute Error: 0.18 tonnes/ha
  R^2 Score: 0.97

--- Example Prediction ---
Inputs: {'crop': 'Wheat', 'rainfall': 520, 'temperature': 22, 'fertilizer': 135, 'soil_ph': 6.5}
Predicted yield: 3.61 tonnes/ha
```

## Customize

- Replace `data/crop_yield_data.csv` with your own real-world dataset (same columns)
- Add more crops, regions, or features (e.g., humidity, sunlight hours, irrigation type)
- Swap the model for `LinearRegression`, `XGBoost`, etc.

## Next Steps / Ideas

- Add a simple web UI with Streamlit or Flask
- Connect to live weather API data
- Add visualization of feature importance
- Deploy as an API endpoint

## License

MIT
