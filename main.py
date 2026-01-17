import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from src.forecaster import DengueForecaster

def main():
    
    # load dataset
    print("Loading dataset...")
    try:
        data = pd.read_csv("data/WeeklyNumberofDengueandDengueHaemorrhagicFeverCases.csv")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # prepare data for cross validation
    forecaster = DengueForecaster()
    
    # use helper to get full processed data with features for cross validation
    X_full, y_full, full_data = forecaster.engineer_features(data)
    
    # time-series cross validation
    print("Running time-series cross validation...")
    tscv = TimeSeriesSplit(n_splits = 5)
    fold = 1
    
    # for comparison as a metric of performance
    benchmark_scores = []
    model_scores = []
    
    plt.figure(figsize = (12, 6))
    
    for train_index, test_index in tscv.split(X_full):
        print(f"Carrying out cross validation on fold {fold}...")
        
        # split data into train-test sets
        X_train, X_test = X_full.iloc[train_index], X_full.iloc[test_index]
        y_train, y_test = y_full.iloc[train_index], y_full.iloc[test_index]
        
        # train decision tree model
        forecaster.model.fit(X_train, y_train)
        model_pred = forecaster.model.predict(X_test)
        
        # calculate mae score
        model_mae = mean_absolute_error(y_test, model_pred)
        model_scores.append(model_mae)
        
        # calculate benchmark mae, using "lag_1_week" as prediction
        benchmark_pred = X_test["lag_1_week"]
        benchmark_mae = mean_absolute_error(y_test, benchmark_pred)
        benchmark_scores.append(benchmark_mae)
        
        print(f"Fold {fold} -- Model MAE: {model_mae:.2f}, Benchmark MAE: {benchmark_mae:.2f}")
        
        # visualisaing the predictions for this fold
        test_dates = full_data.iloc[test_index]["date"]
        label = f"Decision TreePredictions" if fold == 1 else ""
        plt.plot(test_dates, model_pred, label = label, linestyle = '--')
        fold += 1
        
        # reporting performance scores
        avg_model_mae = np.mean(model_scores)
        avg_benchmark_mae = np.mean(benchmark_scores)
        improvement = ((avg_benchmark_mae - avg_model_mae) / avg_benchmark_mae) * 100
    print(f"\nAverage Model MAE over all folds: {avg_model_mae:.2f}")
    print(f"Average Benchmark MAE over all folds: {avg_benchmark_mae:.2f}")
    print(f"Model Improvement over Benchmark: {improvement:.2f}%")
    
    if improvement > 0:
        print("Success! The Decision Tree model outperforms the benchmark")
    else:
        print("The Decision Tree model does not outperform the benchmark!")
    
    # visualising overall results
    plt.plot(full_data["date"], full_data["number"], label = "Actual Numbers", color = "black", alpha = 0.3, linewidth = 2)
    plt.ylabel("Weekly Cases")
    plt.xlabel("Date")
    plt.legend()
    plt.legend()
    plt.grid(True, alpha = 0.3)
    
    # saving the plot
    plt.savefig("results/dengue_cv_results.png")
    print("Cross validation plot saved to results/dengue_cv_results.png")
    
if __name__ == "__main__":
    main()