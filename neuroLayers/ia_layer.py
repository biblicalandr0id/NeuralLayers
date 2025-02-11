class InterpretableAttention(nn.Module):
    """Attention mechanism for interpretability"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_weights = torch.softmax(self.attention(x), dim=1)
        attended = x * attention_weights
        return attended, attention_weights