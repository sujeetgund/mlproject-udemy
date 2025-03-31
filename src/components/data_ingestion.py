import os
from src.exception import CustomException
from src.logger import logger

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'raw_data.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()
    
    def init_data_ingestion(self):
        logger.info("Data Ingestion started")
        try:
            df = pd.read_csv("notebooks/data/stud.csv")
            logger.info("Data loaded successfully")
            
            os.makedirs('artifacts', exist_ok=True)
            logger.info("Artifacts directory created")
            
            df.to_csv(self.config.raw_data_path, index=False)
            logger.info("Raw data saved successfully")
            
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            logger.info("Train test split done")
            
            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)
            logger.info("Train and test data saved successfully")
            
            logger.info("Data Ingestion completed")
            
            return (self.config.train_data_path, self.config.test_data_path)
        except Exception as e:
            raise CustomException(e)


if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.init_data_ingestion()