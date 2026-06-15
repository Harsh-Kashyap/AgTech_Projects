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
├── yield_predictor.py         # main CLI script
├── app.py                      # Streamlit web app
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Run (CLI)

```bash
python yield_predictor.py
```

## Run (Web App - Streamlit)

```bash
streamlit run app.py
```

This opens an interactive browser UI where you can:
- Select a crop type
- Adjust sliders for rainfall, temperature, fertilizer, and soil pH
- Get an instant yield prediction
- See how it compares to the average yield for that crop
- Explore the training data and average yields by crop

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
