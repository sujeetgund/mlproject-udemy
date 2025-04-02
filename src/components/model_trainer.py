import os
from dataclasses import dataclass

from catboost import CatBoostRegressor
from numpy import ndarray
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.logger import logger
from src.exception import CustomException
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_filepath: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
        self.models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "AdaBoost Regressor": AdaBoostRegressor(),
            "KNeighbors Regressor": KNeighborsRegressor(),
            "SVR": SVR(),
            "CatBoost Regressor": CatBoostRegressor(verbose=False),
            "XGBoost Regressor": XGBRegressor(),
        }
        self.model_params = {
            "Decision Tree": {
                "criterion": [
                    "squared_error",
                    "friedman_mse",
                    "absolute_error",
                    "poisson",
                ],
                # 'splitter':['best','random'],
                # 'max_features':['sqrt','log2'],
            },
            "Random Forest": {
                # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                # 'max_features':['sqrt','log2',None],
                "n_estimators": [8, 16, 32, 64, 128, 256]
            },
            "Gradient Boosting": {
                # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                "learning_rate": [0.1, 0.01, 0.05, 0.001],
                "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                # 'criterion':['squared_error', 'friedman_mse'],
                # 'max_features':['auto','sqrt','log2'],
                "n_estimators": [8, 16, 32, 64, 128, 256],
            },
            "Linear Regression": {},
            "KNeighbors Regressor": {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"],
            },
            "SVR": {
                "kernel": ["linear", "poly", "rbf", "sigmoid"],
                "C": [0.01, 0.1, 1, 10, 100],
            },
            "XGBoost Regressor": {
                "learning_rate": [0.1, 0.01, 0.05, 0.001],
                "n_estimators": [8, 16, 32, 64, 128, 256],
            },
            "CatBoost Regressor": {
                "depth": [6, 8, 10],
                "learning_rate": [0.01, 0.05, 0.1],
                "iterations": [30, 50, 100],
            },
            "AdaBoost Regressor": {
                "learning_rate": [0.1, 0.01, 0.5, 0.001],
                # 'loss':['linear','square','exponential'],
                "n_estimators": [8, 16, 32, 64, 128, 256],
            },
        }

    def init_model_trainer(self, train_arr: ndarray, test_arr: ndarray):
        """
        Trains multiple models on the provided training data, evaluates their performance,
        selects the best-performing model, saves it, and calculates its R2 score on the test data.

        Args:
            train_arr (ndarray): A 2D numpy array where the last column represents the target variable
                                 and the remaining columns represent the features for training.
            test_arr (ndarray): A 2D numpy array where the last column represents the target variable
                                and the remaining columns represent the features for testing.

        Raises:
            CustomException: If no model achieves a satisfactory score (R2 score < 0.6).
            CustomException: If any other exception occurs during the process.
        """

        logger.info("Model training started")
        try:
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            logger.info("Running models on the training data")
            training_report = evaluate_models(
                X_train, y_train, X_test, y_test, self.models, self.model_params
            )

            best_model_name = max(training_report, key=lambda k: training_report[k])
            best_model_score = training_report[best_model_name]
            if best_model_score < 0.6:
                raise CustomException("No model is able to give a good score")
            best_model = self.models[best_model_name]
            logger.info(
                f"Best model found: {best_model_name} with R2 score: {best_model_score}"
            )

            logger.info("Saving the best model")
            save_object(
                filepath=self.config.trained_model_filepath,
                obj=best_model,
            )

        except Exception as e:
            raise CustomException(e)
        finally:
            logger.info("Model training completed")
