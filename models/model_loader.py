

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_config):
    name = model_config['name']
    path = model_config['path']
    
    print(f"Loading {name} from {path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    return tokenizer, model

def get_available_models(config):
    return [model['name'] for model in config['llm']['models']]
