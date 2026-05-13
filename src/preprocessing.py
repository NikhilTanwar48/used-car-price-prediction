
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import yaml
from pathlib import Path

class Preprocessor:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
    def build_pipeline(self) -> ColumnTransformer:
        num_features = self.config['features']['numerical']
        cat_ord_features = self.config['features']['categorical_ordinal']
        cat_oh_features = self.config['features']['categorical_onehot']

        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        cat_ord_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])

        cat_oh_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, num_features),
                ('cat_ord', cat_ord_transformer, cat_ord_features),
                ('cat_oh', cat_oh_transformer, cat_oh_features)
            ]
        )
        return preprocessor

    def save(self, transformer: ColumnTransformer):
        path = Path(self.config['paths']['preprocessor_path'])
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(transformer, path)