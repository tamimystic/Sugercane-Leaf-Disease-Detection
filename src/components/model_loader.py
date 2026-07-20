import sys
import torch
from src.exception import CustomException
from src.logger import logging
from src.entity.config_entity import PredictionConfig
from src.components.model_architecture import HybridSEFusion

class ModelLoader:
    def __init__(self, config: PredictionConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def load_model(self):
        try:
            if self.model is None:
                logging.info(f"Loading model on {self.device}")
                num_classes = len(self.config.classes)
                model = HybridSEFusion(num_classes)
                model.load_state_dict(torch.load(self.config.model_path, map_location=self.device))
                model.to(self.device)
                model.eval()
                self.model = model
            return self.model
        except Exception as e:
            raise CustomException(e, sys)
