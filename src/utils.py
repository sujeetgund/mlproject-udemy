import os
import dill

from src.exception import CustomException


def save_object(filepath: str, obj):
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(filepath, 'wb') as file:
            dill.dump(obj, file)
            file.close()
    except Exception as e:
        raise CustomException(e)