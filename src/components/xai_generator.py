import sys
import torch
import numpy as np
import torch.nn.functional as F
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from lime import lime_image
from skimage.segmentation import mark_boundaries
from src.exception import CustomException
from src.logger import logging
from src.utils.common import image_to_base64

class XAIGenerator:
    def __init__(self, model, device, preprocessor, enable_lime=True, lime_samples=150):
        self.model = model
        self.device = device
        self.preprocessor = preprocessor
        self.enable_lime = enable_lime
        self.lime_samples = lime_samples
        
        if self.enable_lime:
            self.lime_explainer = lime_image.LimeImageExplainer()
            
    def generate_gradcam(self, tensor, original_img):
        try:
            logging.info("Generating Grad-CAM++")
            cam = GradCAMPlusPlus(model=self.model, target_layers=[self.model.densenet.features])
            grayscale_cam = cam(input_tensor=tensor, targets=None)[0, :]
            grad_cam_img = show_cam_on_image(original_img.astype(np.float32) / 255.0, grayscale_cam, use_rgb=True)
            return image_to_base64(grad_cam_img)
        except Exception as e:
            raise CustomException(e, sys)

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

    def generate_lime(self, original_img):
        if not self.enable_lime:
            return None
            
        try:
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
            return image_to_base64(lime_img)
        except Exception as e:
            raise CustomException(e, sys)
