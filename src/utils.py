import os
import dill
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logger


def save_object(filepath: str, obj) -> None:
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok=True)

        with open(filepath, "wb") as file:
            dill.dump(obj, file)
            file.close()
    except Exception as e:
        raise CustomException(e)


def evaluate_models(
    X_train, y_train, X_test, y_test, models: dict, models_params: dict
) -> dict:
    results = {}

    for model_name, model in models.items():
        try:
            params = models_params.get(model_name, {})

            gridcv = GridSearchCV(estimator=model, param_grid=params, cv=3, error_score=np.nan)
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
