import pandas as pd
from sklearn.tree import DecisionTreeRegressor

class DengueForecaster:
    def __init__(self, max_depth=5):
        # set max depth to prevent overfitting
        self.model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        
    def engineer_features(self, df):
        data = df.copy()
        
        # cleaning dataset
        data.columns = data.columns.str.lower()
        
        # collectively consider both DHF & Dengue entries in the dataset
        data = data.groupby(["year", "eweek"])["number"].sum().reset_index()
        
        # create a standard format for date (Year + Week number + Day 0 (Sunday))
        data["date_str"] = data["year"].astype(str) + "-" + data["eweek"].astype(str) + "-0"
        data["date"] = pd.to_datetime(data["date_str"], format="%Y-%U-%w")
        
        # sorting chronologically
        data = data.sort_values("date")
        
        # feature engineering
        
        # 1. month
        data["month"] = data["date"].dt.month
        
        # 2. lag
        data["lag_1_week"] = data["number"].shift(1)
        
        # 3. trend (4 week lag)
        data["lag_4_week"] = data["number"].shift(4)
        
        # drop empty rows created from shifting
        data = data.dropna()
        
        features  = ["eweek", "month", "lag_1_week", "lag_4_week"]
        target = "number"
        
        return data[features], data[target], data
    
    def train(self, df):
        X, y, _ = self.engineer_features(df)
        self.model.fit(X, y)
        
    def predict(self, df):
        X, y, processed_data = self.engineer_features(df)
        processed_data["predicted_number"] = self.model.predict(X)
        return processed_data