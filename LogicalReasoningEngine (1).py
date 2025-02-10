import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
import numpy as np

class LogicalRule:
    """Explicit representation of logical rules"""
    def __init__(self, premises: List[str], conclusion: str, weight: float = 1.0):
        self.premises = premises
        self.conclusion = conclusion
        self.weight = weight
        self.activation_history = []  # For interpretability

    def evaluate(self, premise_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Evaluate rule based on premise values"""
        premise_tensors = [premise_values.get(p, torch.tensor(0.0)) for p in self.premises]
        # Logical AND operation on premises
        result = torch.stack(premise_tensors).prod(dim=0)
        self.activation_history.append(float(result.mean()))
        return result * self.weight

class LogicalReasoningLayer(nn.Module):
    """
    Improved logical reasoning layer with explicit rule representation
    and interpretable architecture
    """
    def __init__(self, 
                 input_size: int,
                 num_rules: int,
                 hidden_size: int = 64,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        # Architecture components
        self.premise_encoder = PremiseEncoder(input_size, hidden_size)
        self.rule_engine = RuleEngine(num_rules)
        self.inference_network = InferenceNetwork(hidden_size, dropout_rate)
        
        # Interpretability components
        self.attention = InterpretableAttention(hidden_size)
        self.rule_importance = nn.Parameter(torch.ones(num_rules))
        
        # Uncertainty handling
        self.uncertainty_estimator = UncertaintyEstimator(hidden_size)
        
        # Validation components
        self.consistency_checker = ConsistencyChecker()

    def forward(self, 
                premises: torch.Tensor, 
                rule_inputs: Optional[List[Dict[str, torch.Tensor]]] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with interpretability information
        
        Args:
            premises: Input tensor of premises [batch_size, input_size]
            rule_inputs: Optional explicit rule inputs
            
        Returns:
            conclusions: Tensor of logical conclusions
            interpretation: Dict containing interpretability information
        """
        # 1. Encode premises
        encoded_premises = self.premise_encoder(premises)
        
        # 2. Apply attention for interpretability
        attended_premises, attention_weights = self.attention(encoded_premises)
        
        # 3. Process through rule engine
        rule_outputs = self.rule_engine(attended_premises, rule_inputs)
        
        # 4. Apply inference network
        conclusions = self.inference_network(rule_outputs)
        
        # 5. Estimate uncertainty
        uncertainty = self.uncertainty_estimator(conclusions)
        
        # 6. Check logical consistency
        is_consistent = self.consistency_checker(conclusions)
        
        # Prepare interpretability information
        interpretation = {
            'attention_weights': attention_weights,
            'rule_activations': self.rule_engine.get_rule_activations(),
            'uncertainty': uncertainty,
            'consistency': is_consistent,
            'premise_importance': self.get_premise_importance()
        }
        
        return conclusions, interpretation

class PremiseEncoder(nn.Module):
    """Encodes input premises into latent space"""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

class RuleEngine(nn.Module):
    """Explicit logical rule processing engine"""
    def __init__(self, num_rules: int):
        super().__init__()
        self.num_rules = num_rules
        self.rules: List[LogicalRule] = []
        self.rule_activations = []
    
    def add_rule(self, rule: LogicalRule):
        self.rules.append(rule)
    
    def forward(self, 
                encoded_premises: torch.Tensor,
                rule_inputs: Optional[List[Dict[str, torch.Tensor]]] = None) -> torch.Tensor:
        """Apply logical rules to encoded premises"""
        rule_outputs = []
        self.rule_activations = []
        
        for rule in self.rules:
            if rule_inputs:
                output = rule.evaluate(rule_inputs[0])  # Assuming batch size 1 for simplicity
            else:
                output = self.apply_implicit_rule(encoded_premises, rule)
            rule_outputs.append(output)
            self.rule_activations.append(float(output.mean()))
            
        return torch.stack(rule_outputs, dim=1)
    
    def get_rule_activations(self) -> List[float]:
        return self.rule_activations

class InferenceNetwork(nn.Module):
    """Processes rule outputs into final conclusions"""
    def __init__(self, hidden_size: int, dropout_rate: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Tanh()  # Output in [-1, 1] for logical values
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class InterpretableAttention(nn.Module):
    """Attention mechanism for interpretability"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_weights = torch.softmax(self.attention(x), dim=1)
        attended = x * attention_weights
        return attended, attention_weights

class UncertaintyEstimator(nn.Module):
    """Estimates uncertainty in logical conclusions"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.estimator = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.estimator(x)

class ConsistencyChecker:
    """Checks logical consistency of conclusions"""
    def __call__(self, conclusions: torch.Tensor) -> bool:
        # Check for logical contradictions
        contradiction = torch.any((conclusions > 0.5) & (conclusions < -0.5))
        return not contradiction