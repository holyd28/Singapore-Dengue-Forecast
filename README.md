# Singapore Dengue Forecast Engine
*Python, Docker, Ridge Regression*

- Engineered a containerised time-series forecasting pipeline using ridge regression to model seasonal dengue transmission.
- Implemented lag-feature engineering to capture volatile outbreaks.
- Benchmarked model performance with a 5-fold walk-forward cross-validation.
- Observed that model did not perform well for short-horizon forecasts due to the high autocorrelation inherent in viral transmission rates.

## Project Structure

The repository is organized into a modular pipeline to separate data processing, modeling logic, and execution. To run the project, have the files organised in a folder as such:

```text
singapore-dengue-forecasting/
├── data/
│   └── WeeklyNumberofDengueandDengueHaemorrhagicFeverCases.csv  # Raw data
├── results/
│   └── dengue_cv_results.png    # Generated cross-validation plots
├── src/
│   ├── __init__.py
│   └── forecaster.py            # Feature engineering & ridge regression
├── Dockerfile                   # Container environment
├── main.py                      # Main script for CV & benchmarking
├── requirements.txt             # Python dependencies (pandas, scikit-learn, xgboost)
└── run.sh                       # Shell script to build & run the pipeline
```

Thereafter, you can run the project using Docker or locally with Python (3.9+).

### A. Using Docker
#### 1. Clone the repository
```bash
git clone https://github.com/holyd28/singapore-dengue-forecasting.git
cd singapore-dengue-forecasting
```

#### 2. Build and Run the Container
```bash
./run.sh # to build & execute pipeline
```

### B. Running Locally
#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 2. Execute the Pipeline
```bash
python main.py # This will output the MAE scores and save plots to the /results folder.
```

