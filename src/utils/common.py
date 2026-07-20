import os
import sys
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        import pickle
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        import pickle
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def image_to_base64(img_array: np.ndarray) -> str:
    try:
        img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
        buffered = BytesIO()
        img_pil.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        raise CustomException(e, sys)
