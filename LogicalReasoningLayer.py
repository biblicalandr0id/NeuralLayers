import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import hashlib

class LogicalReasoningLayer(nn.Module):
    def __init__(self, timestamp="2025-02-09 22:20:56", user_login="biblicalandr0id", 
                 input_dim=16, hidden_dim=32, output_dim=8):
        super(LogicalReasoningLayer, self).__init__()
        
        # Constants from our equation
        self.PHI = 1.618033988749895  # Golden ratio
        self.PI = 3.141592653589793
        self.E = 2.718281828459045
        
        # Initialize entropy seed from user data
        self.entropy = self._generate_entropy(timestamp, user_login)
        
        # Premise Vector Weights (ρᵢ) initialized with Fibonacci ratio
        self.premise_weights = self._initialize_premise_weights(hidden_dim)
        
        # Logic Weights (ωᵢ)
        self.logic_weights = self._initialize_logic_weights(hidden_dim)
        
        # Neural Network Layers
        self.phi_function = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            CustomPhiActivation(sigma=self.PHI)
        )
        
        self.cognitive_tensor = CognitiveTensorOperation(hidden_dim)
        
        self.reasoning_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Tanh()  # Maps to [-1, 1] as per θ space
        )
        
        # Learnable parameters for premise and logic weights
        self.premise_scale = nn.Parameter(torch.ones(hidden_dim))
        self.logic_scale = nn.Parameter(torch.ones(hidden_dim))
        
    def _generate_entropy(self, timestamp: str, user_login: str) -> float:
        """Generate deterministic entropy from input parameters"""
        dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
        combined = f"{dt.timestamp()}{user_login}"
        entropy_hash = hashlib.sha256(combined.encode()).hexdigest()
        return int(entropy_hash[:16], 16) / (16 ** 16)
    
    def _initialize_premise_weights(self, size: int) -> torch.Tensor:
        """Initialize ρᵢ weights using Fibonacci sequence"""
        weights = [1.0, self.PHI]
        for i in range(size - 2):
            weights.append(weights[-1] / weights[-2])
        return torch.tensor(weights[:size], dtype=torch.float32)
    
    def _initialize_logic_weights(self, size: int) -> torch.Tensor:
        """Initialize ωᵢ weights based on our equation"""
        weights = torch.zeros(size)
        weights[0] = 1.0        # Base truth
        weights[1] = -1.0       # Contradiction
        weights[2] = 0.5        # Uncertainty
        weights[3:] = 0.382033989  # Probability weight
        return weights
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implementation of ℒ(x) = {Φ × Σ(ρᵢ ⊗ ωᵢ)} → θ
        """
        # Apply Φ function
        phi_out = self.phi_function(x)
        
        # Apply premise and logic weights
        weighted_premises = phi_out * (self.premise_weights * self.premise_scale)
        weighted_logic = phi_out * (self.logic_weights * self.logic_scale)
        
        # Cognitive tensor operation
        tensor_out = self.cognitive_tensor(weighted_premises, weighted_logic)
        
        # Map to reasoning output space θ
        return self.reasoning_output(tensor_out)


class CustomPhiActivation(nn.Module):
    """Custom activation function for Φ = exp(-||x||²/2σ²)"""
    def __init__(self, sigma=1.618033988749895):
        super(CustomPhiActivation, self).__init__()
        self.sigma = sigma
    
    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        return torch.exp(-norm**2 / (2 * self.sigma**2))


class CognitiveTensorOperation(nn.Module):
    """Implementation of cognitive tensor operation ⊗"""
    def __init__(self, hidden_dim):
        super(CognitiveTensorOperation, self).__init__()
        self.hidden_dim = hidden_dim
        
    def forward(self, a, b):
        # a ⊗ b = (a × b) / √(1 + (a² + b²))
        numerator = a * b
        denominator = torch.sqrt(1 + (a**2 + b**2))
        return numerator / (denominator + 1e-7)  # Add epsilon for numerical stability

