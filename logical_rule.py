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
