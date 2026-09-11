# Machine Learning Forecasting of Food Price Volatility in Somalia

An applied machine-learning project that forecasts food commodity prices and converts predicted price levels into a proxy early-warning system for Somalia.

## Project overview

Somalia's food markets are affected by drought, floods, supply-chain disruptions, exchange-rate movements, and fragmented market reporting. This project evaluates whether machine-learning models can forecast food commodity price volatility using historical market prices and exchange-rate data, and whether those forecasts can be translated into practical risk signals for policymakers and humanitarian organizations.

The project compares Random Forest, XGBoost, LightGBM, LSTM, and a hybrid LSTM-XGBoost model. The strongest model is then used to classify predicted prices into four alert levels: Normal, Warning, Alert, and Crisis.

## Objectives

- Prepare and integrate Somalia food-price data with exchange-rate data.
- Engineer time-series, volatility, commodity, market, and regional features.
- Compare tree-based, gradient-boosting, deep-learning, and hybrid models.
- Identify the most important drivers of predicted food prices.
- Create a price-threshold proxy early-warning mechanism.
- Present regional and commodity risks through an interactive dashboard.

## Data sources

- [World Food Programme - Somalia Food Prices](https://data.humdata.org/dataset/wfp-food-prices-for-somalia)
- [World Bank Microdata Library](https://microdata.worldbank.org/index.php/catalog/6155/get-microdata)

The modelling dataset combines historical food-price observations with Somalia's unofficial exchange rate. Raw source data is not redistributed in this repository; users should retrieve it from the original providers and follow their licensing terms.

## Methodology

The project follows the CRISP-DM framework:

1. Business and data understanding
2. Data cleaning and integration
3. Missing-value treatment and outlier analysis
4. Log transformation, encoding, scaling, and feature engineering
5. Feature selection using correlation, multicollinearity checks, F-tests, and mutual information
6. Time-aware model training and evaluation
7. Model comparison and error analysis
8. Threshold-based risk classification and dashboard development

### Models evaluated

- Random Forest Regressor
- XGBoost Regressor
- LightGBM Regressor
- Long Short-Term Memory network (LSTM)
- Hybrid LSTM-XGBoost

### Evaluation metrics

- Coefficient of determination (R²)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)

## Results

| Rank | Model | R² | RMSE | MAE | MAPE |
|---:|---|---:|---:|---:|---:|
| 1 | Random Forest (baseline) | **0.9581** | **0.1903** | **0.1181** | 20.19% |
| 2 | LightGBM (tuned) | 0.9540 | 0.2000 | 0.1284 | **19.85%** |
| 3 | LSTM-XGBoost (baseline) | 0.9539 | 0.2000 | 0.1274 | 20.78% |
| 4 | LightGBM (baseline) | 0.9529 | 0.2020 | 0.1348 | 22.54% |
| 5 | XGBoost (tuned) | 0.9423 | 0.2230 | 0.1461 | 23.21% |
| 6 | XGBoost (baseline) | 0.9418 | 0.2250 | 0.1418 | 22.74% |
| 7 | LSTM-XGBoost (tuned) | 0.9374 | 0.2330 | 0.1513 | 24.71% |
| 8 | Random Forest (tuned) | 0.9228 | 0.2590 | 0.1593 | 24.89% |
| 9 | LSTM (baseline) | 0.8228 | 0.3920 | 0.2710 | 41.73% |
| 10 | LSTM (tuned) | 0.7988 | 0.4180 | 0.2837 | 43.70% |

Random Forest produced the most accurate and stable forecasts. LightGBM achieved the lowest percentage error. Standalone LSTM models performed poorly because Somalia's market observations are sparse and irregular, illustrating that a more complex model is not automatically a better model for fragile datasets.

## Proxy early-warning system

Predicted prices are compared with historical percentile thresholds:

| Risk level | Threshold | Interpretation |
|---|---|---|
| Normal | Below 75th percentile | Price remains within the usual historical range |
| Warning | 75th-90th percentile | Rising price pressure requires monitoring |
| Alert | 90th-95th percentile | Unusual price increase may affect food access |
| Crisis | Above 95th percentile | Extreme price pressure requires urgent attention |

A hierarchical fallback is used when a commodity-season group lacks sufficient history: commodity-season thresholds fall back to commodity-level thresholds, then to market-wide thresholds.

## Key findings

- Rolling price volatility over 3, 6, and 12 months was among the strongest predictive signals.
- Nugaal, Sool, and Bari recorded the highest concentrations of warning and crisis signals.
- Camel meat, wheat flour, white maize, onions, and sorghum were among the most vulnerable commodities.
- Tree-based and boosting models were more reliable than deep learning for sparse, irregular market data.
- Price forecasts can support risk monitoring, but this proxy does not replace a complete food-security early-warning system.

## Technology stack

- Python
- pandas and NumPy
- scikit-learn
- XGBoost
- LightGBM
- TensorFlow/Keras
- Matplotlib and Seaborn
- Interactive dashboarding tools

## Recommended repository structure

```text
somalia-food-price-forecasting/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_comparison.ipynb
│   └── 05_early_warning_system.ipynb
├── src/
│   ├── data_preparation.py
│   ├── features.py
│   ├── train.py
│   └── alerts.py
├── dashboard/
├── reports/
└── images/
```

## Limitations

- Market observations are sparse and irregular across commodities and locations.
- The exchange rate is the only external driver included consistently.
- Climate, conflict, rainfall, transport, and global price variables are not included.
- The alert mechanism is a price-based proxy, not a live operational food-security system.

## Future work

- Add rainfall, drought, conflict, transport, and global commodity-price indicators.
- Validate forecasts using walk-forward testing across multiple forecast horizons.
- Add prediction intervals and uncertainty estimates.
- Develop a reproducible data pipeline and deploy the dashboard as a live application.
- Collaborate with local agencies to validate alerts against field observations.

## Author

**Asma Abdiwali Hassan**  
MSc Data Science and Business Analytics  
[LinkedIn](https://www.linkedin.com/in/asma-hassan-431a37368)

## Responsible use

The alert categories are analytical proxy signals based on predicted price percentiles. They should not be interpreted as official declarations of famine, food insecurity, or humanitarian emergency. Operational decisions require validation with climate, nutrition, conflict, livelihood, and field-assessment data.

## License

Code can be released under the MIT License once the final source files are reviewed. Third-party datasets remain subject to their original providers' terms.
