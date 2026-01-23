# Walmart Sales Forecasting - Complete Analysis & Modeling

## 📊 Project Overview
This project provides a comprehensive end-to-end analysis of Walmart weekly sales data from the Kaggle dataset, implementing both classical time series and machine learning forecasting models with production-ready export capabilities.

## 📁 Project Structure

```
/workspaces/BSc-Computer-Science-Project/
├── walmart_sales_forecasting.ipynb       # Main analysis notebook (88 KB)
├── train.csv                              # Training data (13 MB)
├── features.csv                           # External features (579 KB)
├── stores.csv                             # Store metadata (532 B)
├── test.csv                               # Test data (2.5 MB)
├── sampleSubmission.csv                   # Submission format (2.1 MB)
├── models/                                # Exported models directory
│   ├── best_model_xgboost.pkl            # Trained XGBoost model
│   ├── feature_scaler.pkl                # Feature scaler
│   ├── feature_info.pkl                  # Feature metadata
│   ├── model_metadata.pkl                # Model information
│   ├── preprocessing.py                  # Web app integration module
│   └── INTEGRATION_GUIDE.md              # Deployment documentation
└── ANALYSIS_README.md                    # This file
```

## 🔍 Notebook Structure (12 Sections)

### 1. **Set Up Environment and Extract Data**
   - Import all required libraries (pandas, numpy, scikit-learn, statsmodels, xgboost, matplotlib, seaborn)
   - Extract CSV files from Kaggle dataset archive

### 2. **Import and Load CSV Files**
   - Load train.csv, features.csv, stores.csv, test.csv, and sampleSubmission.csv
   - Verify successful loading with shape and column information

### 3. **Data Inspection and Validation**
   - Examine data types, missing values, and distributions
   - Display first rows and summary statistics for each dataset

### 4. **Preprocessing and Data Merging**
   - Convert Date column to datetime format
   - Merge training data with stores and features on Store and Date keys
   - Handle missing values (fill markdowns with 0, use mean for other features)

### 5. **Feature Engineering**
   - Create time-based features: Year, Month, Week, DayOfWeek, Quarter
   - Encode categorical variables: Type (A/B/C), IsHoliday (True/False)
   - Generate feature statistics and distributions

### 6. **Exploratory Data Analysis**
   - Visualize time series trends and seasonality
   - Plot feature distributions and correlations
   - Identify outliers and anomalies
   - Analyze sales patterns by month, holiday, and store type

### 7. **Train-Test Split by Time**
   - Split data 80-20 based on temporal order (not random shuffling)
   - Prepare feature sets for ML models
   - Ensure validation set mimics real forecasting scenarios

### 8. **Build and Train Time Series Model**
   - Implement ARIMA(1,1,1) model on aggregated daily sales
   - Implement Holt-Winters Exponential Smoothing
   - Generate forecasts and model diagnostics

### 9. **Build and Train Machine Learning Models**
   - Train XGBoost with optimized hyperparameters
   - Train Random Forest regressor
   - Scale features using StandardScaler
   - Analyze feature importance

### 10. **Model Evaluation and Comparison**
   - Calculate MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error)
   - Create comparison visualizations
   - Analyze error patterns (holidays, promotions, etc.)
   - Select best performing model

### 11. **Export Best Model and Preprocessing Artifacts**
   - Serialize best model using joblib
   - Save feature scaler and encoders
   - Export feature metadata and model information

### 12. **Prepare for Web App Integration**
   - Create reusable `SalesPreprocessor` class
   - Generate Flask API example code
   - Provide deployment checklist and integration guide

## 📊 Dataset Information

| File | Rows | Size | Description |
|------|------|------|-------------|
| train.csv | 421,570 | 13 MB | Weekly sales by store and date |
| features.csv | 8,190 | 579 KB | External features (temperature, fuel price, CPI, etc.) |
| stores.csv | 45 | 532 B | Store metadata (type, size, location) |
| test.csv | 115,064 | 2.5 MB | Test data for predictions |
| sampleSubmission.csv | 115,064 | 2.1 MB | Expected output format |

**Date Range**: 2010-02-05 to 2012-10-26 (143 weeks)  
**Total Stores**: 45  
**Features**: Temperature, Fuel Price, CPI, Unemployment, MarkDowns 1-5, IsHoliday

## 🤖 Models Implemented

### Time Series Models
- **ARIMA(1,1,1)**: Classical autoregressive integrated moving average
- **Holt-Winters**: Exponential smoothing with trend and seasonality

### Machine Learning Models
- **XGBoost** ⭐ (Best Performer): Gradient boosting with ~100 trees
- **Random Forest**: Ensemble of decision trees

## 📈 Key Features Used (18 Total)

**Numerical Features:**
- Store, Size, Temperature, Fuel_Price, CPI, Unemployment
- MarkDown1-5 (Promotional discounts)

**Time-Based Features:**
- Year, Month, Week, DayOfWeek, Quarter

**Categorical Features (Encoded):**
- Type_Encoded (Store type: A, B, C)
- IsHoliday (Binary)

## 🔧 Required Libraries

```bash
pip install pandas numpy scikit-learn statsmodels xgboost matplotlib seaborn joblib
```

## 🚀 Quick Start

### Running the Notebook
```python
# Open and run walmart_sales_forecasting.ipynb in Jupyter/VS Code
jupyter notebook walmart_sales_forecasting.ipynb
```

### Using Exported Model
```python
from preprocessing import SalesPreprocessor
import pandas as pd

# Load preprocessor and model
preprocessor = SalesPreprocessor(model_dir='./models')

# Prepare input data
input_data = pd.DataFrame({
    'Store': [1, 2, 3],
    'Date': ['2024-01-15', '2024-01-22', '2024-01-29'],
    'Temperature': [65.5, 70.2, 68.1],
    'Fuel_Price': [3.15, 3.18, 3.12],
    'CPI': [220.5, 220.8, 221.2],
    'Unemployment': [4.5, 4.5, 4.4],
    'MarkDown1': [0, 1000, 500],
    'MarkDown2': [0, 0, 0],
    'MarkDown3': [0, 500, 0],
    'MarkDown4': [0, 0, 0],
    'MarkDown5': [0, 0, 0],
    'IsHoliday': [False, False, False],
    'Type': ['A', 'B', 'C'],
    'Size': [140000, 150000, 160000]
})

# Make predictions
predictions = preprocessor.predict(input_data)
print(f"Predicted sales: {predictions}")
```

## 📊 Expected Results

### Model Performance Metrics
The notebook will calculate and display:
- **Mean Absolute Error (MAE)**: Average prediction error in dollars
- **Root Mean Squared Error (RMSE)**: Penalizes larger errors more heavily
- **R² Score**: Proportion of variance explained (0-1 scale)

### Typical Insights
- XGBoost outperforms traditional time series methods
- Store size is the strongest predictor of sales
- Promotional markdowns have significant impact
- Clear weekly seasonality (Sundays often peak)
- Holiday periods show distinct patterns

## 🌐 Web App Integration

See [models/INTEGRATION_GUIDE.md](models/INTEGRATION_GUIDE.md) for:
- Deployment checklist
- Flask API example
- Feature requirements
- Maintenance guidelines

## 📝 Preprocessing Pipeline

The exported `preprocessing.py` module handles:
1. Date parsing and time feature creation
2. Categorical variable encoding
3. Missing value imputation
4. Feature scaling (StandardScaler)
5. Feature selection and ordering

## ⚙️ Model Files

After running the notebook, the following files are created in `./models/`:

| File | Purpose |
|------|---------|
| best_model_xgboost.pkl | Trained XGBoost model |
| feature_scaler.pkl | StandardScaler for feature normalization |
| feature_info.pkl | Feature columns and encoders |
| model_metadata.pkl | Model version, performance metrics, training date |
| preprocessing.py | Reusable preprocessing class |
| INTEGRATION_GUIDE.md | Web app deployment guide |

## 🔄 Workflow Summary

```
Raw Data (CSV files)
    ↓
Data Loading & Validation
    ↓
Preprocessing & Merging
    ↓
Feature Engineering
    ↓
Exploratory Data Analysis
    ↓
Train-Test Split (Time-based)
    ↓
Model Training (ARIMA, XGBoost, RF)
    ↓
Model Evaluation
    ↓
Best Model Selection
    ↓
Export & Serialization
    ↓
Web App Integration
```

## 📚 References

- **Dataset**: [Walmart Recruiting - Store Sales Forecasting](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting)
- **Libraries**: 
  - [XGBoost Documentation](https://xgboost.readthedocs.io/)
  - [Statsmodels](https://www.statsmodels.org/)
  - [Scikit-learn](https://scikit-learn.org/)

## 👤 Notes

- The notebook uses a time-based split (80% training, 20% validation) to realistically evaluate forecasting performance
- Missing markdown values are filled with 0 (representing no promotion)
- Feature scaling is essential for model convergence and performance
- The exported model is ready for production deployment with Flask/FastAPI

## 📞 Support

For issues or questions:
1. Check the INTEGRATION_GUIDE.md in the models directory
2. Review notebook comments for detailed explanations
3. Verify all CSV files are extracted and in the project root

---

**Project Status**: ✅ Complete  
**Last Updated**: January 23, 2026  
**Notebook Size**: 88 KB  
**Lines of Code**: 2,271
