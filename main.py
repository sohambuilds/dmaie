# main.py

import yaml
import logging
from typing import List
from agents.base_agent import BaseAgent
from agents.debater import Debater
from agents.synthesizer import Synthesizer
from agents.meta_judge import MetaJudge
from agents.bias_checker import BiasChecker
from dmaie_orchestrator import DMAIEOrchestrator
from models.model_loader import load_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def initialize_agents(config: dict) -> List[BaseAgent]:
    num_agents = config['debate']['num_agents']
    gpu_devices = config['gpu']['device_ids']
    models_config = config['llm']['models']
    agent_config = config['agent']
    
    agents = []
    for i in range(num_agents):
        model_config = models_config[i % len(models_config)]
        tokenizer, model = load_model(model_config)
        
        device = f"cuda:{gpu_devices[i % len(gpu_devices)]}"
        
        if i == 0:
            agent = Synthesizer(model_config['name'], tokenizer, model, device)
        elif i == 1:
            agent = MetaJudge(model_config['name'], tokenizer, model, device)
        elif i == 2:
            agent = BiasChecker(model_config['name'], tokenizer, model, device)
        else:
            agent = Debater(model_config['name'], tokenizer, model, device)
        
        agent.set_generation_params(
            temperature=agent_config['temperature'],
            max_tokens=agent_config['max_tokens']
        )
        agents.append(agent)
    
    return agents

def main():
    config = load_config('config.yaml')
    agents = initialize_agents(config)
    orchestrator = DMAIEOrchestrator(agents, config)

    debate_topic = "Should artificial intelligence be regulated by governments?"
    result = orchestrator.run_debate(debate_topic)

    logger.info(f"Debate concluded. Final result: {result}")

if __name__ == "__main__":
    main()