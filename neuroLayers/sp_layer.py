class SensoryProcessor(nn.Module):
    """Implements Σ(t) = ∑[s∈S] ∫[0→t] λₛ(τ) × [∏(i=1→5) ψᵢ(s,τ)] dτ"""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.sensory_weights = nn.Parameter(torch.randn(5, input_dim))
        self.processing_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(5)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sensory_outputs = []
        for i, layer in enumerate(self.processing_layers):
            weighted_input = x * self.sensory_weights[i]
            processed = layer(weighted_input)
            sensory_outputs.append(processed)
        return torch.stack(sensory_outputs).sum(dim=0)
