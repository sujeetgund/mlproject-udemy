import os
from src.exception import CustomException
from src.logger import logger
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw_data.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def init_data_ingestion(self) -> tuple[str, str]:
        """
        Performs the data ingestion process, which includes loading raw data,
        splitting it into training and testing datasets, and saving the results
        to specified file paths.

        Steps:
        1. Loads the dataset from a CSV file.
        2. Creates the necessary directory for storing artifacts.
        3. Saves the raw data to the specified raw data path.
        4. Splits the data into training and testing datasets.
        5. Saves the training and testing datasets to their respective file paths.

        Raises:
            CustomException: If any error occurs during the data ingestion process.

        Returns:
            tuple[str, str]: A tuple containing the file paths of the training
            and testing datasets.
        """

        logger.info("Data Ingestion started")
        try:
            logger.info("Loading dataset")
            # Load the dataset from a CSV file
            df = pd.read_csv("notebooks/data/stud.csv")

            logger.info("Creating artifacts directory")
            # Create the artifacts directory if it doesn't exist
            os.makedirs("artifacts", exist_ok=True)

            logger.info("Saving raw data")
            # Save the raw data to the specified path
            df.to_csv(self.config.raw_data_path, index=False)

            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

            logger.info("Saving train and test data")
            # Save the training and testing datasets to their respective paths
            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)

            return (self.config.train_data_path, self.config.test_data_path)
        except Exception as e:
            raise CustomException(e)
        finally:
            logger.info("Data Ingestion completed")
