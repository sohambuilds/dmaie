# dmaie_orchestrator.py
import torch

import logging
from typing import List, Dict
from agents.base_agent import BaseAgent
from agents.debater import Debater
from agents.synthesizer import Synthesizer
from agents.meta_judge import MetaJudge
from agents.bias_checker import BiasChecker
from utils.role_assignment import assign_roles
from utils.confidence_voting import aggregate_votes
from utils.termination_criteria import should_terminate
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class DMAIEOrchestrator:
    def __init__(self, agents: List[BaseAgent], config: Dict):
        self.agents = agents
        self.config = config
        self.round = 0
        self.debate_history = []
        self.logger = logging.getLogger("DMAIEOrchestrator")

    def clear_gpu_memory(self):
        """Clear GPU memory cache."""
        torch.cuda.empty_cache()
        gc.collect()
        self.logger.info("GPU memory cache cleared")


    def run_debate(self, topic: str):
        self.logger.info(f"Starting debate on topic: {topic}")
        while not should_terminate(self.round, self.debate_history, self.config):
            self.round += 1
            self.logger.info(f"Starting round {self.round}")

            # 1. Dynamic Role Assignment
            roles = assign_roles(self.agents, self.round, self.config)
            for agent, role in zip(self.agents, roles):
                agent.set_role(role)
                self.logger.info(f"Assigned role {role} to agent {agent}")

            # 2. Conduct Debate Round
            round_arguments = self._conduct_round(topic)

            # 3. Synthesize Arguments
            synthesized_arguments = self._synthesize_arguments(round_arguments)

            # 4. Meta-Judgment
            meta_judgment = self._meta_judge(synthesized_arguments, round_arguments)

            # 5. Bias Checking
            bias_report = self._check_bias(round_arguments + [meta_judgment])

            # 6. Cross-Pollination of Ideas
            self._cross_pollinate(round_arguments)

            # 7. Confidence-Weighted Voting
            round_winner = aggregate_votes(round_arguments, self.config)

            # Store round results
            self.debate_history.append({
                'round': self.round,
                'arguments': round_arguments,
                'synthesis': synthesized_arguments,
                'meta_judgment': meta_judgment,
                'bias_report': bias_report,
                'round_winner': round_winner
            })

            self.logger.info(f"Round {self.round} completed. Winner: {round_winner['winner']} with score {round_winner['winner_score']}")
            self.clear_gpu_memory()

        # 8. Final Evaluation
        final_result = self._final_evaluation()
        self.logger.info(f"Debate concluded. Final result: {final_result}")
        return final_result

 
    def _conduct_round(self, topic: str) -> List[Dict]:
        round_arguments = []
        for agent in self.agents:
            if isinstance(agent, Debater):
                argument = agent.process(topic)
                round_arguments.append({
                    'agent': agent,
                    'argument': argument,
                    'confidence': agent.get_confidence()
                })
        return round_arguments

    def _synthesize_arguments(self, round_arguments: List[Dict]) -> str:
        synthesizer = next(agent for agent in self.agents if isinstance(agent, Synthesizer))
        arguments = [arg['argument'] for arg in round_arguments]
        return synthesizer.process(arguments)

    def _meta_judge(self, synthesized_arguments: str, original_arguments: List[Dict]) -> str:
        meta_judge = next(agent for agent in self.agents if isinstance(agent, MetaJudge))
        return meta_judge.process(synthesized_arguments, original_arguments)

    def _check_bias(self, debate_content: List[str]) -> str:
        bias_checker = next(agent for agent in self.agents if isinstance(agent, BiasChecker))
    # Extract the arguments from the dicts and join them as strings
        debate_text = "\n".join([item['argument'] if isinstance(item, dict) else item for item in debate_content])
        return bias_checker.process(debate_text)

    def _cross_pollinate(self, round_arguments: List[Dict]):
        # Find the argument with the highest confidence
        best_argument = max(round_arguments, key=lambda x: x['confidence'])
        
        # Extract key insights (for simplicity, we'll use the full argument)
        key_insight = best_argument['argument']
        
        # Share this insight with all agents
        for agent in self.agents:
            if isinstance(agent, Debater):  # Only update debaters for now
                try:
                    agent.update_knowledge(key_insight)
                except Exception as e:
                    self.logger.error(f"Failed to update knowledge for agent {agent}: {str(e)}")

    
    def _confidence_weighted_voting(self, round_arguments: List[Dict]) -> Dict:
        result = aggregate_votes(round_arguments, self.config)
        
        # Log detailed voting information
        self.logger.info("Confidence-weighted voting results:")
        for arg in round_arguments:
            self.logger.info(f"Agent: {arg['agent'].get_role()}, Confidence: {arg['confidence']}")
        self.logger.info(f"Winner: {result['winner'].get_role()}, Score: {result['winner_score']}")
        
        return result

    def _final_evaluation(self) -> Dict:
        agent_scores = {agent: [] for agent in self.agents if isinstance(agent, Debater)}
        
        for round_data in self.debate_history:
            for arg in round_data['arguments']:
                agent = arg['agent']
                if agent in agent_scores:
                    score = arg['confidence']
                    # Apply simple bias penalty
                    if "bias detected" in round_data['bias_report'].lower():
                        score *= 0.9
                    # Apply meta-judgment adjustment (assuming meta_judgment includes a score)
                    meta_score = float(round_data['meta_judgment'].split(':')[-1])
                    score *= (1 + (meta_score - 0.5) * 0.1)  # Small adjustment based on meta-score
                    agent_scores[agent].append(score)
        
        # Calculate final scores
        final_scores = {agent: sum(scores) / len(scores) for agent, scores in agent_scores.items()}
        winner = max(final_scores, key=final_scores.get)
        
        return {
            'final_scores': final_scores,
            'winner': winner,
            'winner_score': final_scores[winner]
        }