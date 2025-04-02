import os
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logger
from src.utils import save_object

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


@dataclass
class DataTransformationConfig:
    preprocessor_filepath: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_preprocessor(self):
        try:
            numerical_features = ["writing_score", "reading_score"]
            categorical_features = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(sparse_output=False)),
                    # Note: sparse_output=False is used to return a dense array instead of a sparse matrix
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("numerical", numerical_pipeline, numerical_features),
                    ("categorical", categorical_pipeline, categorical_features),
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e)

    def init_data_transformation(
        self, train_data_path: str, test_data_path: str
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """
        Performs data transformation on training and testing datasets, including preprocessing,
        feature engineering, and saving the preprocessor object for future use.

        Args:
            train_data_path (str): Path to the training dataset CSV file.
            test_data_path (str): Path to the testing dataset CSV file.

        Raises:
            CustomException: If any error occurs during the data transformation process.

        Returns:
            tuple[np.ndarray, np.ndarray, str]: A tuple containing:
                - np.ndarray: Transformed training dataset.
                - np.ndarray: Transformed testing dataset.
                - str: Path to the saved preprocessor object.
        """

        logger.info("Data Transformation started")

        try:
            logger.info("Reading train and test data")
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            input_features_train_df = train_df.drop(columns=["math_score"], axis=1)
            target_feature_train_df = train_df["math_score"]

            input_features_test_df = test_df.drop(columns=["math_score"], axis=1)
            target_feature_test_df = test_df["math_score"]

            logger.info("Obtaining preprocessor")
            preprocessor = self.get_preprocessor()

            logger.info("Fitting and transforming train data")
            input_features_train_arr = preprocessor.fit_transform(
                input_features_train_df
            )
            target_feature_train_arr = target_feature_train_df.values.reshape(-1, 1)

            logger.info("Transforming test data")
            input_features_test_arr = preprocessor.transform(input_features_test_df)
            target_feature_test_arr = target_feature_test_df.values.reshape(-1, 1)

            # Above we used reshape to convert the target feature to a 2D array
            # because the model expects a 2D array for the target variable.

            train_arr = np.c_[input_features_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_features_test_arr, target_feature_test_arr]

            logger.info("Saving preprocessor")
            save_object(filepath=self.config.preprocessor_filepath, obj=preprocessor)

            logger.info("Data Transformation completed")

            return (train_arr, test_arr, self.config.preprocessor_filepath)
        except Exception as e:
            raise CustomException(e)
