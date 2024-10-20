# model_manager.py

import torch
from typing import Dict, List
from models.model_loader import load_model

class ModelManager:
    def __init__(self, config: Dict):
        self.config = config
        self.models = {}
        self.current_device = None
        self.load_all_models()

    def load_all_models(self):
        for model_config in self.config['llm']['models']:
            name = model_config['name']
            self.models[name] = load_model(model_config)

    def get_model(self, name: str, device: str):
        if name not in self.models:
            raise ValueError(f"Model {name} not found")
        
        tokenizer, model = self.models[name]  # Unpack tokenizer and model
        if self.current_device != device:
            model.to(device)  # Move only the model to the device
            self.current_device = device
        
        return tokenizer, model  # Return both the tokenizer and model

    def clear_memory(self):
        for model in self.models.values():
            if hasattr(model, 'cpu'):
                model.cpu()
        torch.cuda.empty_cache()
        self.current_device = None