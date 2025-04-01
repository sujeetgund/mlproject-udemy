import os
import dill
from sklearn.metrics import r2_score

from src.exception import CustomException


def save_object(filepath: str, obj) -> None:
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(filepath, 'wb') as file:
            dill.dump(obj, file)
            file.close()
    except Exception as e:
        raise CustomException(e)


def evaluate_models(X_train, y_train, X_test, y_test, models) -> dict:
    results = {}
    
    for model_name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            results[model_name] = score
        except Exception as e:
            raise CustomException(e)
    
    return results