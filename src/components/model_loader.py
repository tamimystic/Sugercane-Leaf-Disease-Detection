import torch
import torch.nn as nn
import timm
from src.exception import CustomException
import sys
from src.entity.config_entity import PredictionConfig
from src.logger import logging

class HybridSEFusion(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Ensure we use pretrained=False here because we will load our own weights.
        self.swin = timm.create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=0, global_pool="avg")
        self.densenet = timm.create_model("densenet121", pretrained=False, num_classes=0, global_pool="avg")
        
        self.ln_swin, self.ln_dense = nn.LayerNorm(1024), nn.LayerNorm(1024)
        self.se = nn.Sequential(
            nn.Linear(2048, 128), 
            nn.ReLU(), 
            nn.Linear(128, 2048), 
            nn.Sigmoid()
        )
        self.head = nn.Sequential(
            nn.Linear(2048, 1024), 
            nn.GELU(), 
            nn.Dropout(0.40), 
            nn.Linear(1024, 512), 
            nn.GELU(), 
            nn.Dropout(0.30), 
            nn.Linear(512, 256), 
            nn.GELU(), 
            nn.Dropout(0.20), 
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        f = torch.cat([self.ln_swin(self.swin(x)), self.ln_dense(self.densenet(x))], dim=1)
        return self.head(f * self.se(f))

class ModelLoader:
    def __init__(self, config: PredictionConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def load_model(self):
        try:
            if self.model is None:
                logging.info(f"Loading model on device: {self.device}")
                num_classes = len(self.config.classes)
                model = HybridSEFusion(num_classes)
                model.load_state_dict(torch.load(self.config.model_path, map_location=self.device))
                model.to(self.device)
                model.eval()
                self.model = model
                logging.info("Model loaded successfully.")
            return self.model
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise CustomException(e, sys)
