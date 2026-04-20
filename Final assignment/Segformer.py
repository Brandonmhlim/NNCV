import torch.nn as nn
from transformers import SegformerForSemanticSegmentation
import os 
import torch

class Model(nn.Module):
    def __init__(self, num_classes=19):
        super().__init__()
        use_local = os.getenv("USE_LOCAL_MODEL", "False") == "True"
        
        if use_local:
            model_path ="./mit-b1"
            local_only = True
            print("ayyyy we on the local model forreal")
        else:
            model_path = "nvidia/mit-b1"
            local_only = False
            print("ayyyy we on the huggingface model forreal")
            
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_path,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            local_files_only=local_only,
        )

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        
        outputs = torch.nn.functional.interpolate(
            outputs.logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )   
        
        return outputs