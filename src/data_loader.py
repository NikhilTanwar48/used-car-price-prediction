import pandas as pd
import logging
from pathlib import Path
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.autoscout_path = Path(self.config['paths']['raw_autoscout'])
        self.ebay_path = Path(self.config['paths']['raw_ebay'])
        self.processed_path = Path(self.config['paths']['processed_data'])

    def _unify_datasets(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """Applies your notebook cleaning logic."""
        logger.info("Unifying AutoScout and eBay datasets...")
        
        df1_clean['Price'] = df1['price']
        df1_clean['Make'] = df1['make_model'].str.split(' ', n=1).str[0]
        df1_clean['Model'] = df1['make_model'].str.split(' ', n=1).str[1]
        df1_clean['Mileage_km'] = df1['km']
        df1_clean['Power_kW'] = df1['hp_kW']
        df1_clean['Fuel_Type'] = df1['Fuel']
        df1_clean['Gearbox'] = df1['Gearing_Type']
        df1_clean['Body_Type'] = df1['body_type']
        df1_clean['Age'] = df1['age']

        df2_clean = pd.DataFrame()
        df2_clean['Price'] = df2['price']
        df2_clean['Make'] = df2['brand'].str.title()
        df2_clean['Model'] = df2['model']
        df2_clean['Mileage_km'] = df2['kilometer']
        df2_clean['Power_kW'] = df2['powerPS'] * 0.735499 
        df2_clean['Fuel_Type'] = df2['fuelType']
        df2_clean['Gearbox'] = df2['gearbox']
        df2_clean['Body_Type'] = df2['vehicleType']
        df2_clean['Age'] = 2024 - df2['yearOfRegistration']

        unified_df = pd.concat([df1_clean, df2_clean], ignore_index=True)
        cols_to_check = ['Model', 'Fuel_Type', 'Gearbox', 'Body_Type']
        
        before_drop = len(unified_df)
        unified_df_clean = unified_df.dropna(subset=cols_to_check).reset_index(drop=True)
        after_drop = len(unified_df_clean)
        
        logger.info(f"Dropped {before_drop - after_drop} rows with missing categoricals.")
        
        unified_df_clean = unified_df_clean[
            (unified_df_clean['Price'] > 100) & (unified_df_clean['Price'] < 300000) &
            (unified_df_clean['Age'] >= 0) & (unified_df_clean['Age'] < 50)
        ]
        return unified_df_clean

    def load_data(self) -> pd.DataFrame:
        if self.processed_path.exists():
            logger.info(f"Loading cached data from {self.processed_path}")
            return pd.read_csv(self.processed_path)
            
        logger.info("Reading raw files...")
        df1 = pd.read_csv(self.autoscout_path)
        df2 = pd.read_csv(self.ebay_path, encoding='latin1')
        
        df = self._unify_datasets(df1, df2)
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.processed_path, index=False)
        return df