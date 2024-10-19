from typing import List, Dict

def should_terminate(current_round: int, debate_history: List[Dict], config: Dict) -> bool:
    """
    Determine if the debate should terminate based on various criteria.
    """
    max_rounds = config.get('max_rounds', 10)
    consensus_threshold = config.get('consensus_threshold', 0.8)
    stagnation_threshold = config.get('stagnation_threshold', 3)
    
    # Check if we've reached the maximum number of rounds
    if current_round >= max_rounds:
        return True
    
    # Check for consensus
    if len(debate_history) > 0:
        last_round = debate_history[-1]
        if 'round_winner' in last_round:
            winner_score = last_round['round_winner'].get('winner_score', 0)
            if winner_score >= consensus_threshold:
                return True
    
    # Check for stagnation (no significant change in scores for several rounds)
    if len(debate_history) >= stagnation_threshold:
        recent_scores = [round_data['round_winner'].get('winner_score', 0) 
                         for round_data in debate_history[-stagnation_threshold:]]
        if max(recent_scores) - min(recent_scores) < 0.1:  # Assuming scores are between 0 and 1
            return True
    
    return False