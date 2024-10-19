# debater.py

from .base_agent import BaseAgent
import torch
import logging

class Debater(BaseAgent):
    def __init__(self, model_name, tokenizer, model, device):
        super().__init__(model_name, tokenizer, model, device)
        self.set_role("Debater")
        self.logger = logging.getLogger(f"Debater_{id(self)}")

    def process(self, input_text):
        """
        Process the input text to generate a debate argument.
        """
        self.logger.info("Generating debate argument")
        
        # Incorporate additional knowledge if available
        if hasattr(self, 'additional_knowledge'):
            additional_context = "\n".join(self.additional_knowledge)
            input_text = f"{input_text}\n\nAdditional context to consider:\n{additional_context}"
        
        prompt = self._construct_prompt(input_text)
        response = self.generate_response(prompt)
        self.evaluate_confidence(input_text, response)
        return response

    def update_knowledge(self, new_information):
        """
        Update the debater's knowledge with new information.
        """
        if not hasattr(self, 'additional_knowledge'):
            self.additional_knowledge = []
        self.additional_knowledge.append(new_information)
        self.logger.info(f"Debater updated with new knowledge: {new_information[:50]}...")


    def _construct_prompt(self, input_text):
        """
        Construct a prompt for the debater based on the input text.
        """
        return f"""As a debater, your task is to present a well-reasoned argument on the following topic:

{input_text}

Please provide a clear and concise argument, considering the following points:
1. State your main position clearly.
2. Provide supporting evidence or examples.
3. Address potential counterarguments.
4. Conclude with a strong summary of your position.

Your response:"""

    def evaluate_confidence(self, input_text, output_text):
        """
        Evaluate the confidence of the debater in its argument.
        This is a simplified version and can be enhanced with more sophisticated methods.
        """
        # Simple confidence evaluation based on output length and presence of key phrases
        confidence = min(len(output_text) / 500, 1.0)  # Normalize by expected length
        key_phrases = ["because", "therefore", "evidence suggests", "research shows"]
        confidence += sum(phrase in output_text.lower() for phrase in key_phrases) * 0.1
        confidence = min(confidence, 1.0)
        
        self.set_confidence(confidence)
        self.logger.debug(f"Confidence evaluated: {confidence}")

    def rebut(self, original_input, opponent_argument):
        """
        Generate a rebuttal to an opponent's argument.
        """
        self.logger.info("Generating rebuttal")
        rebuttal_prompt = f"""As a debater, your task is to rebut the following argument on this topic:

Topic: {original_input}

Opponent's argument: {opponent_argument}

Please provide a clear and concise rebuttal, considering the following points:
1. Identify the main points of the opponent's argument.
2. Challenge the evidence or reasoning presented.
3. Provide counter-evidence or alternative interpretations.
4. Reinforce your original position with new supporting points.

Your rebuttal:"""

        rebuttal = self.generate_response(rebuttal_prompt)
        self.evaluate_confidence(rebuttal_prompt, rebuttal)
        return rebuttal