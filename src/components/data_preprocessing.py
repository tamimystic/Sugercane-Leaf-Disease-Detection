import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.exception import CustomException
import sys
from src.entity.config_entity import PredictionConfig
from src.logger import logging

class DataPreprocessor:
    def __init__(self, config: PredictionConfig):
        self.config = config
        self.val_transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
            ToTensorV2()
        ])
    
    def preprocess_image(self, image_bytes: bytes):
        try:
            logging.info("Starting image preprocessing")
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image.")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            c = min(h, w)
            y, x = (h - c) // 2, (w - c) // 2
            
            img_cropped = img[y:y+c, x:x+c]
            img_resized = cv2.resize(img_cropped, (self.config.image_size, self.config.image_size), interpolation=cv2.INTER_CUBIC)
            
            tensor = self.val_transform(image=img_resized)["image"]
            tensor = tensor.unsqueeze(0)
            
            logging.info("Image preprocessing completed successfully")
            return tensor, img_resized
            
        except Exception as e:
            logging.error(f"Error in image preprocessing: {str(e)}")
            raise CustomException(e, sys)
