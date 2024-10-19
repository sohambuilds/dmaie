# meta_judge.py

from .base_agent import BaseAgent
import logging

class MetaJudge(BaseAgent):
    def __init__(self, model_name, tokenizer, model, device):
        super().__init__(model_name, tokenizer, model, device)
        self.set_role("MetaJudge")
        self.logger = logging.getLogger(f"MetaJudge_{id(self)}")

    def process(self, synthesis, original_arguments):
        """
        Process a synthesis and original arguments to provide a meta-judgment.
        """
        self.logger.info("Generating meta-judgment")
        prompt = self._construct_prompt(synthesis, original_arguments)
        judgment = self.generate_response(prompt)
        self.evaluate_confidence(prompt, judgment)
        return judgment

    def _construct_prompt(self, synthesis, original_arguments):
        """
        Construct a prompt for the meta-judge based on the synthesis and original arguments.
        """
        return f"""As a meta-judge, your task is to evaluate the quality of the debate and provide guidance for future rounds. 
Consider the following synthesis and the original arguments:

Synthesis: {synthesis}

Original Arguments:
{original_arguments}

Please provide a meta-judgment that:
1. Assesses the overall quality and progress of the debate
2. Identifies any logical fallacies or weak points in the arguments
3. Suggests areas where the debate could be improved or expanded
4. Provides guidance for the next round of the debate

Your meta-judgment:"""

    def evaluate_confidence(self, input_text, output_text):
        """
        Evaluate the confidence of the meta-judge in its judgment.
        """
        # Confidence based on output length and presence of key evaluation phrases
        confidence = min(len(output_text) / 400, 1.0)  # Normalize by expected length
        key_phrases = ["quality of debate", "logical fallacy", "weak point", "improvement", "guidance"]
        confidence += sum(phrase in output_text.lower() for phrase in key_phrases) * 0.1
        confidence = min(confidence, 1.0)
        
        self.set_confidence(confidence)
        self.logger.debug(f"Confidence evaluated: {confidence}")