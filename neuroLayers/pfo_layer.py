class ProcessesFinalOutput(nn.Module):
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
