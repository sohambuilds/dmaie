# Dynamic Multi-Agent Iterative Evaluation (DMAIE) Framework

## Project Overview

The Dynamic Multi-Agent Iterative Evaluation (DMAIE) Framework is an innovative approach to leveraging Large Language Models (LLMs) in a multi-agent debate setting. Our goal is to create a system where multiple AI agents, each powered by an LLM, can engage in sophisticated debates, evaluations, and collaborative problem-solving.

## Key Features

- Multiple AI agents with dynamic role assignment
- Iterative debate process with synthesis and meta-judgment
- Bias checking and cross-pollination of ideas
- Confidence-weighted voting system
- Adaptive termination criteria

## How It Works

1. **Initialization**: Multiple agents are initialized, each with a specific LLM (e.g., GPT-3, BLOOM, Mistral).
2. **Dynamic Role Assignment**: Agents are assigned roles such as Debater, Synthesizer, MetaJudge, and BiasChecker.
3. **Debate Process**: Agents engage in multiple rounds of debate on a given topic.
4. **Synthesis and Meta-Judgment**: After each round, arguments are synthesized and evaluated.
5. **Bias Checking**: A dedicated agent checks for potential biases in the debate.
6. **Cross-Pollination**: Strong arguments are shared among agents to improve overall debate quality.
7. **Voting**: A confidence-weighted voting system determines the strongest arguments.
8. **Termination**: The debate continues until termination criteria are met.

## Key Components

- `DMAIEOrchestrator`: Manages the overall debate process.
- `BaseAgent`: Abstract base class for all agent types.
- `Debater`, `Synthesizer`, `MetaJudge`, `BiasChecker`: Specific agent implementations.
- `ModelManager`: Handles loading and device management for LLMs.

## Usage Example

```python
config = load_config('config.yaml')
model_manager = ModelManager(config)
agents = initialize_agents(config, model_manager)
orchestrator = DMAIEOrchestrator(agents, config, model_manager)
result = orchestrator.run_debate("Should artificial intelligence be regulated by governments?")

## Architecture Overview

The DMAIE framework is built on a modular architecture that leverages multiple Large Language Models (LLMs) to create a sophisticated multi-agent debate and evaluation system. The core components are:

1. Model Manager
2. Agent Classes
3. Orchestrator
4. Utility Functions

### 1. Model Manager

The `ModelManager` class is responsible for loading and managing LLMs across multiple GPU devices. It implements lazy loading and device switching to optimize memory usage.

```python
class ModelManager:
    def __init__(self, config: Dict):
        self.config = config
        self.models = {}
        self.current_device = None

    def get_model(self, name: str, device: str):
        if name not in self.models:
            self.models[name] = load_model(self.config['llm']['models'][name])
        tokenizer, model = self.models[name]
        if self.current_device != device:
            model.to(device)
            self.current_device = device
        return tokenizer, model

    def clear_memory(self):
        for _, model in self.models.values():
            model.cpu()
        torch.cuda.empty_cache()
        self.current_device = None
```

### 2. Agent Classes

The framework defines several agent types, all inheriting from a `BaseAgent` abstract base class:

```python
class BaseAgent(ABC):
    def __init__(self, model_name, tokenizer, model, device):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.role = None
        self.confidence = 0.0
        self.generation_params = {'temperature': 0.7, 'max_tokens': 500}

    @abstractmethod
    def process(self, input_text): pass

    @abstractmethod
    def evaluate_confidence(self, input_text, output_text): pass

    def generate_response(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.generation_params['max_tokens'],
                temperature=self.generation_params['temperature'],
                num_return_sequences=1,
                top_p=0.9,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
```

Specific agent types (Debater, Synthesizer, MetaJudge, BiasChecker) implement the abstract methods with role-specific logic.

### 3. Orchestrator

The `DMAIEOrchestrator` class manages the entire debate process:

```python
class DMAIEOrchestrator:
    def __init__(self, agents: List[BaseAgent], config: Dict, model_manager: ModelManager):
        self.agents = agents
        self.config = config
        self.model_manager = model_manager
        self.round = 0
        self.debate_history = []

    def run_debate(self, topic: str):
        while not should_terminate(self.round, self.debate_history, self.config):
            self.round += 1
            roles = assign_roles(self.agents, self.round, self.config)
            for agent, role in zip(self.agents, roles):
                agent.set_role(role)

            round_arguments = self._conduct_round(topic)
            synthesized_arguments = self._synthesize_arguments(round_arguments)
            meta_judgment = self._meta_judge(synthesized_arguments, round_arguments)
            bias_report = self._check_bias(round_arguments + [meta_judgment])
            self._cross_pollinate(round_arguments)
            round_winner = aggregate_votes(round_arguments, self.config)

            self.debate_history.append({
                'round': self.round,
                'arguments': round_arguments,
                'synthesis': synthesized_arguments,
                'meta_judgment': meta_judgment,
                'bias_report': bias_report,
                'round_winner': round_winner
            })

            self.model_manager.clear_memory()

        return self._final_evaluation()
```

### 4. Utility Functions

Several utility functions support the debate process:

- `assign_roles`: Dynamically assigns roles to agents each round.
- `aggregate_votes`: Implements confidence-weighted voting.
- `should_terminate`: Checks termination criteria.

## Current Technical Challenges

The primary challenge we're facing is a device mismatch error during model forward pass:

```
Error: Expected all tensors to be on the same device, but found at least two devices, cuda:1 and cuda:0! (when checking argument for argument weight in method wrapper_CUDA__native_layer_norm)
```

This error occurs despite our checks showing all model parameters on the expected device. Potential causes include:

1. Hidden tensors or buffers on unexpected devices
2. Issues with the accelerate library's device management
3. Potential problems in the layer normalization implementation

We've implemented detailed logging to track device assignments:

```python
def generate_response(self, input_text, max_length=100):
    try:
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        print(f"BaseAgent generating response. Self device: {self.device}")
        print(f"BaseAgent model device: {next(self.model.parameters()).device}")
        print(f"Inputs device: {inputs['input_ids'].device}")
        
        for name, param in self.model.named_parameters():
            if param.device != self.device:
                print(f"Mismatch found: {name} is on {param.device}, expected {self.device}")
        
        with torch.no_grad():
            outputs = self.model.generate(...)
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        self.logger.error(f"Error generating response: {str(e)}")
        return None
```

However, this logging hasn't revealed the source of the cuda:1 device in the error message, suggesting the issue may lie in a part of the model or process not currently being inspected.

## Next Steps

1. Implement more comprehensive device checking, including buffers and submodules.
2. Investigate potential interactions between different model components during the forward pass.
3. Analyze the layer normalization implementation for potential device inconsistencies.
4. Consider implementing a custom autograd function to track tensor device changes during forward and backward passes.
5. Explore alternative multi-GPU strategies, such as model parallelism or pipeline parallelism, to mitigate device management issues.