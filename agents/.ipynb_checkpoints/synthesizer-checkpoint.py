# synthesizer.py

from .base_agent import BaseAgent
import logging

class Synthesizer(BaseAgent):
    def __init__(self, model_name, tokenizer, model, device):
        super().__init__(model_name, tokenizer, model, device)
        self.set_role("Synthesizer")
        self.logger = logging.getLogger(f"Synthesizer_{id(self)}")
        self.additional_knowledge = []

    def process(self, arguments):
        """
        Process a list of arguments to generate a synthesis.
        """
        self.logger.info("Synthesizing arguments")
        
        # Incorporate additional knowledge if available
        if self.additional_knowledge:
            arguments.append("Additional context: " + " ".join(self.additional_knowledge))
        
        prompt = self._construct_prompt(arguments)
        synthesis = self.generate_response(prompt)
        self.evaluate_confidence(prompt, synthesis)
        return synthesis

    def _construct_prompt(self, arguments):
        """
        Construct a prompt for the synthesizer based on the input arguments.
        """
        arguments_text = "\n".join([f"Argument {i+1}: {arg}" for i, arg in enumerate(arguments)])
        return f"""As a synthesizer, your task is to combine and summarize the following arguments:

{arguments_text}

Please provide a coherent synthesis that:
1. Identifies the main points of agreement and disagreement
2. Highlights the strongest arguments from each perspective
3. Suggests potential areas of compromise or further exploration
4. Presents a balanced overview of the debate so far

Your synthesis:"""

    def evaluate_confidence(self, input_text, output_text):
        """
        Evaluate the confidence of the synthesizer in its synthesis.
        """
        # Simple confidence evaluation based on output length and coverage of input arguments
        confidence = min(len(output_text) / 300, 1.0)  # Normalize by expected length
        for arg in input_text.split("Argument")[1:]:
            if any(phrase in output_text.lower() for phrase in arg.lower().split()):
                confidence += 0.1
        confidence = min(confidence, 1.0)
        
        self.set_confidence(confidence)
        self.logger.debug(f"Confidence evaluated: {confidence}")

    def update_knowledge(self, new_information):
        """
        Update the synthesizer's knowledge with new information.
        """
        self.additional_knowledge.append(new_information)
        self.logger.info(f"Synthesizer updated with new knowledge: {new_information[:50]}...")