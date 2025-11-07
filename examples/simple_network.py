"""
Simple Network Example

Demonstrates basic usage of NeuralLayers framework with minimal setup.
"""

import torch
from logicalbrain_network import UnifiedBrainNetwork


def main():
    """Run a simple network example"""

    print("=" * 70)
    print("NeuralLayers - Simple Network Example")
    print("=" * 70)

    # Configuration
    input_dim = 1024
    hidden_dim = 512
    batch_size = 16

    print(f"\nConfiguration:")
    print(f"  Input dimension:  {input_dim}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Batch size:       {batch_size}")

    # Initialize model
    print(f"\n{'Step 1: Initializing model':<40}", end="")
    model = UnifiedBrainNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=4
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ ({total_params:,} parameters)")

    # Create sample input
    print(f"{'Step 2: Creating sample input':<40}", end="")
    x = torch.randn(batch_size, input_dim)
    print(f"✅ Shape: {tuple(x.shape)}")

    # Forward pass
    print(f"{'Step 3: Running forward pass':<40}", end="")
    model.eval()
    with torch.no_grad():
        output = model(x)
    print("✅")

    # Display results
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)

    print(f"\n{'Output':<30} {'Shape':<20} {'Mean':<12} {'Std':<12}")
    print("-" * 70)

    # Main output
    out = output['output']
    print(f"{'Final output':<30} {str(tuple(out.shape)):<20} "
          f"{out.mean().item():<12.4f} {out.std().item():<12.4f}")

    # System state
    state = output['system_state']

    # Membrane potential
    V = state['V']
    print(f"{'Membrane potential (V)':<30} {str(tuple(V.shape)):<20} "
          f"{V.mean().item():<12.2f} {V.std().item():<12.2f}")

    # Neurotransmitters
    NT = state['NT']
    print(f"{'Neurotransmitters (NT)':<30} {str(tuple(NT.shape)):<20} "
          f"{NT.mean().item():<12.2f} {NT.std().item():<12.2f}")

    # Calcium
    Ca = state['Ca']
    print(f"{'Calcium (Ca)':<30} {str(tuple(Ca.shape)):<20} "
          f"{Ca.mean().item():<12.2f} {Ca.std().item():<12.2f}")

    # ATP
    ATP = state['ATP']
    print(f"{'ATP energy':<30} {str(tuple(ATP.shape)):<20} "
          f"{ATP.mean().item():<12.2f} {ATP.std().item():<12.2f}")

    # Glial state
    g = state['g']
    print(f"{'Glial state (g)':<30} {str(tuple(g.shape)):<20} "
          f"{g.mean().item():<12.4f} {g.std().item():<12.4f}")

    # Cognitive state
    Psi = state['Psi']
    print(f"{'Cognitive state (Ψ)':<30} {str(tuple(Psi.shape)):<20} "
          f"{Psi.mean().item():<12.4f} {Psi.std().item():<12.4f}")

    # Truth values
    tau = state['tau']
    print(f"{'Truth values (τ)':<30} {str(tuple(tau.shape)):<20} "
          f"{tau.mean().item():<12.4f} {tau.std().item():<12.4f}")

    # Reasoning momentum
    omega = state['omega']
    print(f"{'Reasoning momentum (ω)':<30} {str(tuple(omega.shape)):<20} "
          f"{omega.mean().item():<12.4f} {omega.std().item():<12.4f}")

    print("\n" + "=" * 70)
    print("Biological Constraints Check")
    print("=" * 70)

    # Check constraints
    V_min, V_max = V.min().item(), V.max().item()
    V_valid = -70.0 <= V_min and V_max <= 40.0
    print(f"  Membrane potential: [{V_min:.2f}, {V_max:.2f}] mV  {'✅' if V_valid else '❌'}")

    NT_min, NT_max = NT.min().item(), NT.max().item()
    NT_valid = NT_min >= 0.0 and NT_max <= 10.0
    print(f"  Neurotransmitters:  [{NT_min:.2f}, {NT_max:.2f}] μM   {'✅' if NT_valid else '❌'}")

    Ca_min, Ca_max = Ca.min().item(), Ca.max().item()
    Ca_valid = Ca_min >= 0.0 and Ca_max <= 1000.0
    print(f"  Calcium:            [{Ca_min:.2f}, {Ca_max:.2f}] nM  {'✅' if Ca_valid else '❌'}")

    ATP_min, ATP_max = ATP.min().item(), ATP.max().item()
    ATP_valid = ATP_min >= 1000.0 and ATP_max <= 10000.0
    print(f"  ATP:                [{ATP_min:.2f}, {ATP_max:.2f}] μM {'✅' if ATP_valid else '❌'}")

    g_min, g_max = g.min().item(), g.max().item()
    g_valid = g_min >= 0.0 and g_max <= 1.0
    print(f"  Glial state:        [{g_min:.4f}, {g_max:.4f}]      {'✅' if g_valid else '❌'}")

    tau_min, tau_max = tau.min().item(), tau.max().item()
    tau_valid = tau_min >= 0.0 and tau_max <= 1.0
    print(f"  Truth values:       [{tau_min:.4f}, {tau_max:.4f}]      {'✅' if tau_valid else '❌'}")

    all_valid = V_valid and NT_valid and Ca_valid and ATP_valid and g_valid and tau_valid

    print("\n" + "=" * 70)
    if all_valid:
        print("✅ All biological constraints satisfied!")
    else:
        print("⚠️  Some constraints violated - check model configuration")
    print("=" * 70)

    print("\n✨ Example completed successfully!")


if __name__ == '__main__':
    main()
