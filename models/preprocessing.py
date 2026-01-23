
"""
Walmart Sales Forecasting - Preprocessing Module
This module provides functions to preprocess data for the trained forecasting models.
"""

import pandas as pd
import numpy as np
import joblib
import pickle
from datetime import datetime
from pathlib import Path

class SalesPreprocessor:
    """Handles data preprocessing for sales forecasting models."""

    def __init__(self, models_dir='./models'):
        """
        Initialize preprocessor with saved artifacts.

        Args:
            models_dir: Directory containing saved models and preprocessing artifacts
        """
        self.models_dir = models_dir

        # Load preprocessing artifacts
        self.scaler = joblib.load(f'{models_dir}/scaler.joblib')
        self.type_encoder = joblib.load(f'{models_dir}/type_encoder.joblib')

        with open(f'{models_dir}/feature_info.pkl', 'rb') as f:
            self.feature_info = pickle.load(f)

        with open(f'{models_dir}/model_metadata.pkl', 'rb') as f:
            self.metadata = pickle.load(f)

        self.feature_columns = self.feature_info['feature_names']

    def preprocess_input(self, data):
        """
        Preprocess input data for model prediction.

        Args:
            data: Dictionary or DataFrame with raw input data

        Returns:
            Preprocessed feature array ready for model prediction
        """
        if isinstance(data, dict):
            data = pd.DataFrame([data])

        data = data.copy()

        # Convert Date to datetime if present
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])

            # Create time-based features
            data['Year'] = data['Date'].dt.year
            data['Month'] = data['Date'].dt.month
            data['Week'] = data['Date'].dt.isocalendar().week
            data['DayOfWeek'] = data['Date'].dt.dayofweek
            data['Quarter'] = data['Date'].dt.quarter

        # Encode Type column
        if 'Type' in data.columns:
            data['Type_Encoded'] = self.type_encoder.transform(data['Type'].fillna('A'))

        # Convert IsHoliday to integer
        if 'IsHoliday' in data.columns:
            data['IsHoliday'] = data['IsHoliday'].astype(int)

        # Fill missing values
        for col in self.feature_info['numerical_features']:
            if col in data.columns:
                data[col].fillna(data[col].mean(), inplace=True)

        # Select only required features
        X = data[self.feature_columns].copy()

        # Scale features
        X_scaled = self.scaler.transform(X)

        return X_scaled

    def predict(self, data, model=None):
        """
        Make prediction using preprocessed data.

        Args:
            data: Raw input data (dict or DataFrame)
            model: Model object. If None, loads the best model.

        Returns:
            Predicted sales value
        """
        X_scaled = self.preprocess_input(data)

        if model is None:
            best_model_name = self.metadata['best_model']
            if best_model_name == 'XGBoost':
                model = joblib.load(f'{self.models_dir}/best_model_xgboost.joblib')
            else:
                model = joblib.load(f'{self.models_dir}/best_model_rf.joblib')

        prediction = model.predict(X_scaled)
        return prediction[0] if len(prediction) == 1 else prediction

    def batch_predict(self, data, model=None):
        """
        Make predictions for multiple samples.

        Args:
            data: DataFrame with multiple rows
            model: Model object. If None, loads the best model.

        Returns:
            Array of predictions
        """
        X_scaled = self.preprocess_input(data)

        if model is None:
            best_model_name = self.metadata['best_model']
            if best_model_name == 'XGBoost':
                model = joblib.load(f'{self.models_dir}/best_model_xgboost.joblib')
            else:
                model = joblib.load(f'{self.models_dir}/best_model_rf.joblib')

        return model.predict(X_scaled)

    def get_model_info(self):
        """Return metadata about the trained model."""
        return self.metadata


# Example usage:
# from preprocessing import SalesPreprocessor
# 
# preprocessor = SalesPreprocessor('./models')
# 
# # Single prediction
# sample_data = {
#     'Store': 1,
#     'Date': '2024-01-15',
#     'Type': 'A',
#     'Size': 151315,
#     'Temperature': 45.5,
#     'Fuel_Price': 2.9,
#     'CPI': 215.5,
#     'Unemployment': 5.5,
#     'IsHoliday': 0,
#     'MarkDown1': 100,
#     'MarkDown2': 0,
#     'MarkDown3': 0,
#     'MarkDown4': 0,
#     'MarkDown5': 0
# }
# 
# prediction = preprocessor.predict(sample_data)
# print(f"Predicted Sales: ${prediction:,.2f}")
