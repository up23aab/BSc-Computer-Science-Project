# Walmart Sales Forecasting

BSc Computer Science Final Year Project focused on forecasting Walmart weekly sales using machine learning.

## Overview
This project predicts weekly sales across Walmart stores and departments by combining:
- Historical sales data
- Store metadata
- Economic and seasonal indicators

The notebook implements an end-to-end pipeline with three models:
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

## Project Structure
- `walmart_sales_forecasting.ipynb`: Main analysis notebook (data prep, EDA, feature engineering, modeling, evaluation)
- `train.csv`: Historical weekly sales (training data)
- `test.csv`: Test rows for prediction/submission format
- `features.csv`: Economic and markdown-related variables by store/date
- `stores.csv`: Store metadata (type and size)
- `sampleSubmission.csv`: Example Kaggle submission format

Generated outputs from the notebook include:
- `walmart_merged.csv`
- `figure_1_temperature_analysis.png`
- `figure_2_department_analysis.png`
- `figure_3_model_comparison.png`
- `figure_4_best_model_evaluation.png`
- `figure_5_feature_importance.png`
- `figure_6_department_mape.png`

## Methodology
The notebook is organized into three parts:

1. Data loading, merging, and cleaning
- Merge `train.csv`, `stores.csv`, and `features.csv`
- Resolve duplicate holiday columns
- Fill missing values for key numeric fields
- Save unified dataset as `walmart_merged.csv`

2. Exploratory Data Analysis (EDA) and Feature Engineering
- Temperature vs sales analysis
- Department-level sales analysis
- Temporal feature extraction from date (year, month, week, day of week, quarter)
- Encode categorical store type
- Feature selection (drop high-missing markdown fields and low-signal fields)

3. Model Training, Evaluation, and Comparison
- Time-based 80/20 split to avoid data leakage
- Train Linear Regression, Random Forest, and XGBoost
- Evaluate using MAE, RMSE, and R2
- Analyze residuals and feature importance
- Compare department-level performance with MAPE

## Tech Stack
- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- xgboost
- Jupyter Notebook

## Setup
1. Clone the repository:
```bash
git clone https://github.com/up23aab/BSc-Computer-Science-Project
cd BSc-Computer-Science-Project
```

2. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

## Run
Launch Jupyter and open the notebook:
```bash
jupyter notebook walmart_sales_forecasting.ipynb
```

Run cells from top to bottom to reproduce:
- Data preparation
- Visualizations
- Model training and evaluation
- Output figures and summary tables

## Notes
- Keep all CSV files in the project root so the notebook can find them.
- The notebook includes a fallback path for a different execution environment; for normal GitHub usage, local root files are sufficient.

## Future Improvements
- Hyperparameter tuning with cross-validation or Bayesian optimization
- Time-series-aware validation strategies (rolling window backtesting)
- Ensemble/blending methods for improved forecast stability
- Model tracking and experiment logging (for reproducibility)

## Author
BSc Computer Science Final Year Project
