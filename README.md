# Singapore Dengue Forecast Engine
*Python, Docker, Ridge Regression*

- Engineered a containerised time-series forecasting pipeline using ridge regression to model seasonal dengue transmission.
- Implemented lag-feature engineering to capture volatile outbreaks.
- Benchmarked model performance with a 5-fold walk-forward cross-validation.
- Observed that model did not perform well for short-horizon forecasts due to the high autocorrelation inherent in viral transmission rates.
