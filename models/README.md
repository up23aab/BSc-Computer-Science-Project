# Walmart Sales Forecasting - Model Files

This directory contains all necessary files for integrating the sales forecasting model into your web application.

## Files Description

### Model Files
- `best_model_xgboost.joblib` - The trained XGBoost model (if it's the best model)
- `best_model_rf.joblib` - The trained Random Forest model (if it's the best model)

### Preprocessing Artifacts
- `scaler.joblib` - StandardScaler for feature normalization
- `type_encoder.joblib` - LabelEncoder for store type encoding
- `feature_info.pkl` - Feature names and metadata
- `model_metadata.pkl` - Model performance metrics and configuration

### Integration Module
- `preprocessing.py` - Reusable preprocessing functions for web app

## Quick Start for Web App

### 1. Install Requirements
```bash
pip install pandas numpy scikit-learn xgboost joblib
```

### 2. Load and Use in Your Web App
```python
from preprocessing import SalesPreprocessor

# Initialize preprocessor
preprocessor = SalesPreprocessor('./models')

# Prepare input data
sample_input = {
    'Store': 1,
    'Date': '2024-01-15',
    'Type': 'A',
    'Size': 151315,
    'Temperature': 45.5,
    'Fuel_Price': 2.9,
    'CPI': 215.5,
    'Unemployment': 5.5,
    'IsHoliday': 0,
    'MarkDown1': 100,
    'MarkDown2': 0,
    'MarkDown3': 0,
    'MarkDown4': 0,
    'MarkDown5': 0
}

# Make prediction
prediction = preprocessor.predict(sample_input)
print(f"Predicted Weekly Sales: ${prediction:,.2f}")
```

### 3. Batch Predictions
```python
import pandas as pd

# Load multiple records
data = pd.read_csv('new_sales_data.csv')

# Get predictions for all records
predictions = preprocessor.batch_predict(data)
```

## Feature Requirements

Required input features for predictions:
- `Store`: Store ID (1-45)
- `Date`: Date in format YYYY-MM-DD
- `Type`: Store type (A, B, or C)
- `Size`: Store size (numeric)
- `Temperature`: Average temperature
- `Fuel_Price`: Fuel price
- `CPI`: Consumer Price Index
- `Unemployment`: Unemployment rate
- `IsHoliday`: Binary flag (0 or 1)
- `MarkDown1` through `MarkDown5`: Markdown amounts (0 if not applicable)

## Model Performance

See `model_metadata.pkl` for:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Mean Absolute Percentage Error (MAPE)

## Important Notes

1. All features must be provided for predictions
2. The Date field is used to automatically generate temporal features (Year, Month, Week, DayOfWeek, Quarter)
3. The preprocessor automatically handles scaling and encoding
4. For production use, ensure consistent data validation and error handling

## Support

For questions about model usage or preprocessing, refer to the main Jupyter notebook:
`walmart_sales_forecasting.ipynb`
