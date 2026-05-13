import joblib
import pandas as pd
import numpy as np
import yaml

class Predictor:
    def __init__(self):
        with open("config/config.yaml", 'r') as f:
            self.config = yaml.safe_load(f)
        self.model = joblib.load(self.config['paths']['model_path'])
        self.preprocessor = joblib.load(self.config['paths']['preprocessor_path'])
        self.fe = joblib.load("models/feature_engineering.joblib")

    def predict(self, input_data: dict) -> dict:
        input_data['Make'] = str(input_data['Make']).title().strip()
        input_data['Model'] = str(input_data['Model']).title().strip()
        
        if 'Power_HP' in input_data:
            hp_value = float(input_data['Power_HP'])
            input_data['Power_kW'] = hp_value * 0.735499
        
        df = pd.DataFrame([input_data])
        
        cols_to_convert = ['Mileage_km', 'Age', 'Power_kW']
        for col in cols_to_convert:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col])

        df_fe = self.fe.transform(df)
        X_proc = self.preprocessor.transform(df_fe)
        
        prediction = self.model.predict(X_proc)[0]
        std = prediction * 0.12 

        return {
            "predicted_price": round(float(prediction), 2),
            "confidence_low": round(float(max(0, prediction - std)), 2),
            "confidence_high": round(float(prediction + std), 2)
        }