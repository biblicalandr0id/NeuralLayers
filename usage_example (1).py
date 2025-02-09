# Example usage of the LogicalReasoningLayer

def test_logical_reasoning():
    # Initialize layer
    layer = LogicalReasoningLayer(
        timestamp="2025-02-09 22:20:56",
        user_login="biblicalandr0id",
        input_dim=16,
        hidden_dim=32,
        output_dim=8
    )
    
    # Create test input
    batch_size = 4
    x = torch.randn(batch_size, 16)  # 16-dimensional input
    
    # Forward pass
    output = layer(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output values (should be in [-1, 1]):\n{output}")
    
    # Test logical operations
    # Example: Testing simple syllogism
    syllogism = torch.tensor([
        [1.0, 1.0, 0.0],  # All humans are mortal
        [1.0, 1.0, 0.0],  # Socrates is human
        [0.0, 0.0, 0.0]   # Therefore... (should be close to 1.0)
    ]).unsqueeze(0)
    
    conclusion = layer(syllogism.float())
    print(f"\nSyllogism conclusion: {conclusion[0][0]}")  # Should be close to 1.0

if __name__ == "__main__":
    test_logical_reasoning()