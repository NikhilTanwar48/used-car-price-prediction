import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

class FeatureEngineer:
    def __init__(self):
        self.model_means = {}
        self.global_mean = 0

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['km_per_year'] = df['Mileage_km'] / df['Age'].replace(0, 1)
        
        if 'Price' in df.columns:
            self.global_mean = df['Price'].mean()
            df['model_avg_price'] = np.nan
            
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            for train_idx, val_idx in kf.split(df):
                means = df.iloc[train_idx].groupby('Model')['Price'].mean()
                df.loc[df.index[val_idx], 'model_avg_price'] = df.loc[df.index[val_idx], 'Model'].map(means)
            
            self.model_means = df.groupby('Model')['Price'].mean().to_dict()
            df['model_avg_price'] = df['model_avg_price'].fillna(self.global_mean)
        
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['km_per_year'] = df['Mileage_km'] / pd.to_numeric(df['Age']).replace(0, 1)
        
        df['model_avg_price'] = df['Model'].map(self.model_means).fillna(self.global_mean)
        
        return df