import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from configs.configs import model_weight, device

def load_model():
    processor = TrOCRProcessor.from_pretrained(model_weight)
    model = VisionEncoderDecoderModel.from_pretrained(model_weight)
    model.to(device)
    model.eval()
    
    return model,processor
