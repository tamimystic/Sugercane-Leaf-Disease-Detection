import torch
import torch.nn.functional as F
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from lime import lime_image
from skimage.segmentation import mark_boundaries
from src.components.data_preprocessing import DataPreprocessor
from src.components.model_loader import ModelLoader
from src.entity.config_entity import PredictionConfig
from src.exception import CustomException
import sys
from src.logger import logging

class PredictionPipeline:
    def __init__(self, enable_lime=True, lime_samples=150):
        self.config = PredictionConfig()
        self.preprocessor = DataPreprocessor(self.config)
        self.model_loader = ModelLoader(self.config)
        
        self.model = self.model_loader.load_model()
        self.device = self.model_loader.device
        
        self.enable_lime = enable_lime
        self.lime_samples = lime_samples
        if self.enable_lime:
            self.lime_explainer = lime_image.LimeImageExplainer()

    def image_to_base64(self, img_array: np.ndarray) -> str:
        img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
        buffered = BytesIO()
        img_pil.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _predict_fn_for_lime(self, images):
        batch_size = 16
        all_preds = []
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i+batch_size]
            batch = torch.stack([self.preprocessor.val_transform(image=img)["image"] for img in batch_imgs], dim=0).to(self.device)
            with torch.no_grad():
                preds = F.softmax(self.model(batch), dim=1).cpu().numpy()
                all_preds.append(preds)
        return np.vstack(all_preds)

    def predict(self, image_bytes: bytes) -> dict:
        try:
            logging.info("Starting prediction and XAI pipeline")
            tensor, original_img = self.preprocessor.preprocess_image(image_bytes)
            tensor = tensor.to(self.device)
            
            with torch.no_grad():
                output = self.model(tensor)
                probs = F.softmax(output, dim=1)[0].cpu().numpy()
            
            # Get all 15 classes and their confidences, sorted from highest to lowest
            top_indices = np.argsort(probs)[::-1]
            all_classes = [{"class": self.config.classes[idx], "confidence": float(probs[idx] * 100)} for idx in top_indices]
            predicted_class = all_classes[0]["class"]
            
            logging.info("Generating Grad-CAM++")
            cam = GradCAMPlusPlus(model=self.model, target_layers=[self.model.densenet.features])
            grayscale_cam = cam(input_tensor=tensor, targets=None)[0, :]
            grad_cam_img = show_cam_on_image(original_img.astype(np.float32) / 255.0, grayscale_cam, use_rgb=True)
            
            lime_b64 = None
            if self.enable_lime:
                logging.info("Generating LIME Boundaries")
                explanation = self.lime_explainer.explain_instance(
                    original_img, 
                    self._predict_fn_for_lime, 
                    top_labels=1, 
                    hide_color=0, 
                    num_samples=self.lime_samples, 
                    random_seed=42
                )
                temp, mask = explanation.get_image_and_mask(
                    explanation.top_labels[0], 
                    positive_only=True, 
                    num_features=5, 
                    hide_rest=False
                )
                lime_img = mark_boundaries(temp / 255.0, mask)
                lime_b64 = self.image_to_base64(lime_img)
            
            original_b64 = self.image_to_base64(original_img.astype(np.float32) / 255.0)
            grad_cam_b64 = self.image_to_base64(grad_cam_img)
            
            result = {
                "predicted_class": predicted_class,
                "all_classes": all_classes,
                "images": {
                    "original": f"data:image/jpeg;base64,{original_b64}",
                    "grad_cam": f"data:image/jpeg;base64,{grad_cam_b64}",
                    "lime": f"data:image/jpeg;base64,{lime_b64}" if lime_b64 else None
                }
            }
            
            logging.info("Prediction and XAI generation successful.")
            return result
            
        except Exception as e:
            logging.error(f"Error in prediction pipeline: {str(e)}")
            raise CustomException(e, sys)
