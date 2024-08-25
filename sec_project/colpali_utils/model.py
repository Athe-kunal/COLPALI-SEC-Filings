import os

import torch
from transformers import AutoProcessor

from colpali_engine.models.paligemma_colbert_architecture import ColPali


def load_model(MODEL_NAME:str="google/paligemma-3b-mix-448",device:str="cuda"):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "vidore/colpali"
    token = os.environ.get("HF_TOKEN")
    model = ColPali.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map=device, token=token
    ).eval()
    model.load_adapter(model_name)
    processor = AutoProcessor.from_pretrained(model_name, token=token)
    device = model.device
    
    return model, device, processor