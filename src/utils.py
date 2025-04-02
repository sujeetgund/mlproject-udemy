import os
import dill
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logger


def save_object(filepath: str, obj) -> None:
    """
    Saves a Python object to a file using dill serialization.

    Args:
        filepath (str): The path where the object should be saved.
        obj (Any): The Python object to be serialized and saved.

    Raises:
        CustomException: If an error occurs during the saving process.
    """

    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok=True)

        with open(filepath, "wb") as file:
            dill.dump(obj, file)
            file.close()
    except Exception as e:
        raise CustomException(e)


def load_object(filepath: str):
    """
    Load a serialized object from a file.

    Args:
        filepath (str): The path to the file containing the serialized object.

    Raises:
        CustomException: If an error occurs during file loading or deserialization.

    Returns:
        Any: The deserialized object loaded from the file.
    """
    try:
        with open(filepath, "rb") as file:
            return dill.load(file)
    except Exception as e:
        raise CustomException(e)


def evaluate_models(
    X_train, y_train, X_test, y_test, models: dict, models_params: dict
) -> dict:
    """
    Evaluates multiple machine learning models using GridSearchCV for hyperparameter tuning
    and calculates the R-squared score for each model on the test dataset.

    Args:
        X_train (array-like or DataFrame): Training feature dataset.
        y_train (array-like or Series): Training target dataset.
        X_test (array-like or DataFrame): Testing feature dataset.
        y_test (array-like or Series): Testing target dataset.
        models (dict): A dictionary where keys are model names (str) and values are model instances.
        models_params (dict): A dictionary where keys are model names (str) and values are parameter grids
                              (dict) for hyperparameter tuning.

    Raises:
        CustomException: If an error occurs during model evaluation or hyperparameter tuning.

    Returns:
        dict: A dictionary where keys are model names (str) and values are the R-squared scores (float)
              of the respective models on the test dataset.
    """

    results = {}

    for model_name, model in models.items():
        try:
            params = models_params.get(model_name, {})

            gridcv = GridSearchCV(
                estimator=model, param_grid=params, cv=3, error_score=np.nan
            )
            gridcv.fit(X_train, y_train)

            logger.info(f"Best parameters for {model_name}: {gridcv.best_params_}")
            model.set_params(**gridcv.best_params_)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            results[model_name] = score
        except Exception as e:
            raise CustomException(e)

    return results
