import os
from dataclasses import dataclass

from catboost import CatBoostRegressor
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
            "AdaBoost": AdaBoostRegressor(),
            "KNN": KNeighborsRegressor(),
            "SVR": SVR(),
            "CatBoost": CatBoostRegressor(verbose=False),
            "XGBoost": XGBRegressor(),
        }

    def init_model_trainer(self, train_arr, test_arr):
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
                X_train, y_train, X_test, y_test, self.models
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

            y_pred = best_model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            logger.info(f"R2 score of the best model on test data: {r2}")

            logger.info("Model training completed")

            return r2

        except Exception as e:
            raise CustomException(e)
