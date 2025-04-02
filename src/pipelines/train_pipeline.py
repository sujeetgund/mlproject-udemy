from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.logger import logger
from src.exception import CustomException


class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def train(self):
        """
        Executes the training pipeline which includes data ingestion,
        data transformation, and model training.

        The method performs the following steps:
        1. Initiates the data ingestion process to fetch training and testing datasets.
        2. Transforms the ingested data into a format suitable for model training.
        3. Trains the model using the transformed data and evaluates its performance.
        Logs are generated at each step to track the progress of the pipeline.

        Raises:
            CustomException: If any error occurs during the execution of the training pipeline.
        """

        logger.info("Training pipeline started")
        try:
            # Data Ingestion
            train_data_path, test_data_path = self.data_ingestion.init_data_ingestion()

            # Data Transformation
            train_arr, test_arr, _ = self.data_transformation.init_data_transformation(
                train_data_path=train_data_path, test_data_path=test_data_path
            )

            # Model Training
            self.model_trainer.init_model_trainer(train_arr, test_arr)

        except Exception as e:
            raise CustomException(e)
        finally:
            logger.info("Training pipeline finished")
            

if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.train()
