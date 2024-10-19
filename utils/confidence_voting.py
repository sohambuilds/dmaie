from typing import List, Dict

def aggregate_votes(round_arguments: List[Dict], config: Dict) -> Dict:
    """
    Aggregate votes based on confidence scores of the arguments.
    """
    total_confidence = sum(arg['confidence'] for arg in round_arguments)
    
    if total_confidence == 0:
        # If all confidences are 0, return equal weights
        weight = 1.0 / len(round_arguments)
        return {arg['agent']: weight for arg in round_arguments}
    
    # Calculate weighted votes
    weighted_votes = {}
    for arg in round_arguments:
        agent = arg['agent']
        confidence = arg['confidence']
        weighted_votes[agent] = confidence / total_confidence
    
    # Find the winner
    winner = max(weighted_votes, key=weighted_votes.get)
    
    return {
        'votes': weighted_votes,
        'winner': winner,
        'winner_score': weighted_votes[winner]
    }