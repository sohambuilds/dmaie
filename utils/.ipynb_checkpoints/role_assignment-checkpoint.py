import random
from typing import List, Dict
from agents.base_agent import BaseAgent

def assign_roles(agents: List[BaseAgent], current_round: int, config: Dict) -> List[str]:
    """
    Dynamically assign roles to agents based on the current round and configuration.
    """
    num_agents = len(agents)
    num_debaters = config.get('num_debaters', max(2, num_agents - 3))
    
    roles = ['Debater'] * num_debaters
    roles.extend(['Synthesizer', 'MetaJudge', 'BiasChecker'])
    
    # Ensure we have enough roles for all agents
    while len(roles) < num_agents:
        roles.append('Debater')
    
    # Shuffle roles to ensure randomness
    random.shuffle(roles)
    
    # Ensure that agents don't get the same role in consecutive rounds
    if current_round > 1:
        previous_roles = [agent.get_role() for agent in agents]
        while any(prev == curr for prev, curr in zip(previous_roles, roles)):
            random.shuffle(roles)
    
    return roles