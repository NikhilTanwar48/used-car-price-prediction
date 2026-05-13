import pandas as pd
import numpy as np
import logging
import joblib
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

def train():
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    df = DataLoader().load_data()
    
    if len(df) > 50000:
        logger.info("Sampling dataset down to 50,000 rows for faster training...")
        df = df.sample(50000, random_state=42)

    fe = FeatureEngineer()
    df = fe.fit_transform(df)
    config['features']['numerical'].extend(['km_per_year', 'model_avg_price'])

    pp = Preprocessor()
    pipeline = pp.build_pipeline()
    
    X = df.drop(columns=[config['features']['target']])
    y = df[config['features']['target']]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['model_params']['test_size'], random_state=42
    )

    X_train_proc = pipeline.fit_transform(X_train)
    X_test_proc = pipeline.transform(X_test)
    pp.save(pipeline)

    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
    model.fit(X_train_proc, y_train)
    
    preds = model.predict(X_test_proc)
    logger.info(f"RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.2f}")
    logger.info(f"MAE: {mean_absolute_error(y_test, preds):.2f}")
    logger.info(f"R2: {r2_score(y_test, preds):.4f}")

    Path(config['paths']['model_path']).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config['paths']['model_path'])
    joblib.dump(fe, "models/feature_engineering.joblib")
    logger.info("Models saved successfully.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()