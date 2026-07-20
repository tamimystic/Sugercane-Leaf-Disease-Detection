import torch
import torch.nn as nn
import timm

class HybridSEFusion(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
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
