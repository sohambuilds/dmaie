# base_agent.py

import torch
import logging
from abc import ABC, abstractmethod

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class BaseAgent(ABC):
    def __init__(self, model_name, tokenizer, model, device):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.role = None
        self.confidence = 0.0
        self.generation_params = {
            'temperature': 0.7,
            'max_tokens': 500
        }
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{id(self)}")

    @abstractmethod
    def process(self, input_text):
        pass
    def set_generation_params(self, temperature=None, max_tokens=None):
        if temperature is not None:
            self.generation_params['temperature'] = temperature
        if max_tokens is not None:
            self.generation_params['max_tokens'] = max_tokens

    def generate_response(self, input_text, max_length=100):
        try:
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.generation_params['max_tokens'],
                    temperature=self.generation_params['temperature'],
                    num_return_sequences=1,
                    top_p=0.9,
                )
        
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response

        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return None

    def set_role(self, role):
        self.role = role
        self.logger.info(f"Role set to: {role}")

    def get_role(self):
        return self.role

    def set_confidence(self, confidence):
        self.confidence = max(0.0, min(1.0, confidence))
        self.logger.debug(f"Confidence set to: {self.confidence}")

    def get_confidence(self):
        return self.confidence

    def update_knowledge(self, new_information):
        """
        Placeholder method for updating agent knowledge.
        TODO: Implement specific knowledge update mechanisms for each agent type.
        """
        self.logger.info(f"Knowledge update received: {new_information[:50]}...")
        pass

    @abstractmethod
    def evaluate_confidence(self, input_text, output_text):
        pass

    def to(self, device):
        self.device = device
        self.model.to(device)
        self.logger.info(f"Model moved to device: {device}")

    def __str__(self):
        return f"{self.__class__.__name__}(model={self.model_name}, role={self.role}, device={self.device})"