# bias_checker.py

from .base_agent import BaseAgent
import logging

class BiasChecker(BaseAgent):
    def __init__(self, model_name, tokenizer, model, device):
        super().__init__(model_name, tokenizer, model, device)
        self.set_role("BiasChecker")
        self.logger = logging.getLogger(f"BiasChecker_{id(self)}")

    def process(self, debate_content):
        """
        Process the entire debate content to check for biases.
        """
        self.logger.info("Checking for biases")
        prompt = self._construct_prompt(debate_content)
        bias_report = self.generate_response(prompt)
        self.evaluate_confidence(prompt, bias_report)
        return bias_report

    def _construct_prompt(self, debate_content):
        """
        Construct a prompt for the bias checker based on the debate content.
        """
        return f"""As a bias checker, your task is to analyze the following debate content for potential biases:

{debate_content}

Please provide a bias report that:
1. Identifies any cognitive biases present in the arguments (e.g., confirmation bias, anchoring bias)
2. Highlights instances of logical fallacies (e.g., ad hominem, straw man arguments)
3. Detects any potential demographic or ideological biases
4. Suggests ways to mitigate these biases in future rounds of the debate

Your bias report:"""

    def evaluate_confidence(self, input_text, output_text):
        """
        Evaluate the confidence of the bias checker in its report.
        """
        # Confidence based on output length and identification of specific biases
        confidence = min(len(output_text) / 500, 1.0)  # Normalize by expected length
        bias_types = ["cognitive bias", "logical fallacy", "demographic bias", "ideological bias"]
        confidence += sum(bias in output_text.lower() for bias in bias_types) * 0.1
        confidence = min(confidence, 1.0)
        
        self.set_confidence(confidence)
        self.logger.debug(f"Confidence evaluated: {confidence}")