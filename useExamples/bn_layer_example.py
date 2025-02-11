# Example usage
if __name__ == "__main__":
    # Initialize network
    brain = BrainNetwork()
    
    # Create sample input (batch_size=1, 6 sensory inputs)
    sensory_input = torch.rand(1, 6)
    
    # Forward pass
    outputs, state = brain(sensory_input)
    
    print("Network State:")
    print(f"Membrane Potential: {outputs['membrane_potential'].item():.2f} mV")
    print(f"Calcium Level: {outputs['calcium'].item():.2f} nM")
    print(f"ATP Level: {outputs['ATP'].item():.2f} μM")
    print(f"Neurotransmitter Level: {outputs['neurotransmitter'].item():.2f} μM")
    print("\nOutputs:")
    print(f"Motor Output Shape: {outputs['motor'].shape}")
    print(f"Autonomic Output Shape: {outputs['autonomic'].shape}")
    print(f"Cognitive Output Shape: {outputs['cognitive'].shape}")