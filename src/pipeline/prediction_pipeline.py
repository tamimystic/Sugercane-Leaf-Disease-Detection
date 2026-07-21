import sys
import torch
import numpy as np
import torch.nn.functional as F
from src.components.data_preprocessing import DataPreprocessor
from src.components.model_loader import ModelLoader
from src.components.xai_generator import XAIGenerator
from src.entity.config_entity import PredictionConfig
from src.exception import CustomException
from src.logger import logging
from src.utils.common import image_to_base64

class PredictionPipeline:
    def __init__(self, lime_samples=40):
        self.config = PredictionConfig()
        
        # Initialize components
        self.preprocessor = DataPreprocessor(self.config)
        self.model_loader = ModelLoader(self.config)
        
        # Load model and device
        self.model = self.model_loader.load_model()
        self.device = self.model_loader.device
        
        # Initialize XAI Generator
        self.xai_generator = XAIGenerator(
            model=self.model,
            device=self.device,
            preprocessor=self.preprocessor,
            lime_samples=lime_samples
        )

    def predict(self, image_bytes: bytes, enable_lime: bool = False) -> dict:
        try:
            logging.info("Starting prediction pipeline")
            
            # Data Preprocessing
            tensor, original_img = self.preprocessor.preprocess_image(image_bytes)
            tensor = tensor.to(self.device)
            
            # Inference
            with torch.no_grad():
                output = self.model(tensor)
                probs = F.softmax(output, dim=1)[0].cpu().numpy()
            
            # Process results
            top_indices = np.argsort(probs)[::-1]
            all_classes = [{"class": self.config.classes[idx], "confidence": float(probs[idx] * 100)} for idx in top_indices]
            predicted_class = all_classes[0]["class"]
            
            # Generate XAI Visualizations
            grad_cam_b64 = self.xai_generator.generate_gradcam(tensor, original_img)
            lime_b64 = self.xai_generator.generate_lime(original_img, enable=enable_lime)
            original_b64 = image_to_base64(original_img.astype(np.float32) / 255.0)
            
            result = {
                "predicted_class": predicted_class,
                "all_classes": all_classes,
                "images": {
                    "original": f"data:image/jpeg;base64,{original_b64}",
                    "grad_cam": f"data:image/jpeg;base64,{grad_cam_b64}",
                    "lime": f"data:image/jpeg;base64,{lime_b64}" if lime_b64 else None
                }
            }
            
            logging.info("Prediction successful")
            return result
            
        except Exception as e:
            raise CustomException(e, sys)
