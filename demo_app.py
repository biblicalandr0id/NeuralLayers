"""
NeuralLayers Interactive Demo

A simple interactive demonstration of the NeuralLayers framework.
Can be run standalone or with Streamlit for web interface.

Usage:
    python demo_app.py              # Command-line interface
    streamlit run demo_app.py       # Web interface (requires streamlit)
"""

import sys
import torch
import numpy as np

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from logicalbrain_network import UnifiedBrainLogicNetwork
from brain_network_implementation import BrainNetwork
from consciousness_layers import ConsciousnessEmergence
from LogicalReasoningLayer import LogicalReasoningLayer
from umi_layer import UMI_Network
from utils import Config, DeviceManager


def create_model_demo():
    """Create and demonstrate the unified model."""

    if STREAMLIT_AVAILABLE:
        st.title("🧠 NeuralLayers Interactive Demo")
        st.markdown("### Unified Neural-Logical Network Framework")

        # Sidebar configuration
        st.sidebar.header("⚙️ Configuration")
        input_dim = st.sidebar.slider("Input Dimension", 128, 2048, 1024, 128)
        hidden_dim = st.sidebar.slider("Hidden Dimension", 256, 4096, 2048, 256)
        output_dim = st.sidebar.slider("Output Dimension", 64, 1024, 512, 64)
        batch_size = st.sidebar.slider("Batch Size", 1, 64, 8, 1)

        # Device selection
        use_cuda = st.sidebar.checkbox("Use CUDA (if available)",
                                       value=torch.cuda.is_available())

        st.sidebar.markdown("---")
        st.sidebar.info(f"💾 Total parameters: ~{(input_dim * hidden_dim + hidden_dim * output_dim) / 1e6:.1f}M")

    else:
        print("=" * 70)
        print(" " * 20 + "🧠 NeuralLayers Demo")
        print("=" * 70)
        print("\n[Configuration]")
        input_dim, hidden_dim, output_dim, batch_size = 1024, 2048, 512, 8
        use_cuda = torch.cuda.is_available()
        print(f"  Input: {input_dim}, Hidden: {hidden_dim}, Output: {output_dim}")
        print(f"  Batch size: {batch_size}")
        print(f"  Device: {'CUDA' if use_cuda else 'CPU'}")

    # Setup device
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')

    # Create model
    if STREAMLIT_AVAILABLE:
        with st.spinner("Initializing model..."):
            model = UnifiedBrainLogicNetwork(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim
            ).to(device)
        st.success("✅ Model initialized!")
    else:
        print("\n[Initializing Model...]")
        model = UnifiedBrainLogicNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        ).to(device)
        print("✅ Model ready!")

    # Generate input
    x = torch.randn(batch_size, input_dim).to(device)

    # Forward pass
    if STREAMLIT_AVAILABLE:
        if st.button("🚀 Run Forward Pass"):
            with st.spinner("Processing..."):
                with torch.no_grad():
                    output = model(x)

                st.success("✅ Forward pass complete!")

                # Display results
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Output Shape", str(tuple(output['output'].shape)))
                    st.metric("Mean Activation", f"{output['output'].mean().item():.4f}")

                with col2:
                    st.metric("Membrane Potential (mV)",
                             f"{output['membrane_potential'].mean().item():.2f}")
                    st.metric("Truth Values",
                             f"{output['truth_values'].mean().item():.4f}")

                # System state
                st.subheader("📊 System State")
                state = output['system_state']

                state_data = []
                for component in ['V', 'NT', 'Ca', 'ATP', 'g', 'Ψ', 'τ', 'ω']:
                    if component in state:
                        mean_val = state[component].mean().item()
                        std_val = state[component].std().item()
                        state_data.append({
                            'Component': component,
                            'Mean': f"{mean_val:.4f}",
                            'Std': f"{std_val:.4f}"
                        })

                import pandas as pd
                st.dataframe(pd.DataFrame(state_data), hide_index=True)
    else:
        print("\n[Running Forward Pass...]")
        with torch.no_grad():
            output = model(x)

        print("✅ Complete!\n")
        print("[Results]")
        print(f"  Output shape: {output['output'].shape}")
        print(f"  Mean activation: {output['output'].mean().item():.4f}")
        print(f"  Membrane potential: {output['membrane_potential'].mean().item():.2f} mV")
        print(f"  Truth values: {output['truth_values'].mean().item():.4f}")

        print("\n[System State]")
        state = output['system_state']
        for component in ['V', 'NT', 'Ca', 'ATP', 'g', 'Ψ', 'τ', 'ω']:
            if component in state:
                mean_val = state[component].mean().item()
                print(f"  {component}: {mean_val:8.4f}")


def brain_demo():
    """Demonstrate brain network."""

    if STREAMLIT_AVAILABLE:
        st.header("🧠 Brain Network Simulation")

        st.markdown("""
        Multi-modal sensory processing with biophysical dynamics:
        - Membrane potential (-70 to +40 mV)
        - Calcium signaling (0-1000 nM)
        - ATP metabolism (1000-10000 μM)
        - Neurotransmitter kinetics (0-10 μM)
        """)

        # Sensory input sliders
        st.subheader("📥 Sensory Input")
        col1, col2, col3 = st.columns(3)

        with col1:
            visual = st.slider("👁️ Visual", 0.0, 1.0, 0.5)
            auditory = st.slider("👂 Auditory", 0.0, 1.0, 0.3)
        with col2:
            tactile = st.slider("🖐️ Tactile", 0.0, 1.0, 0.2)
            olfactory = st.slider("👃 Olfactory", 0.0, 1.0, 0.1)
        with col3:
            gustatory = st.slider("👅 Gustatory", 0.0, 1.0, 0.1)
            proprioceptive = st.slider("🦵 Proprioceptive", 0.0, 1.0, 0.4)

        if st.button("🔬 Simulate Brain"):
            brain = BrainNetwork()
            sensory_input = torch.tensor([[visual, auditory, tactile, olfactory,
                                          gustatory, proprioceptive]])

            outputs, state = brain(sensory_input)

            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("⚡ Membrane Potential",
                         f"{outputs['membrane_potential'].item():.2f} mV")
                st.metric("💊 Calcium",
                         f"{outputs['calcium'].item():.2f} nM")
            with col2:
                st.metric("🔋 ATP",
                         f"{outputs['ATP'].item():.2f} μM")
                st.metric("💉 Neurotransmitter",
                         f"{outputs['neurotransmitter'].item():.2f} μM")
    else:
        print("\n" + "=" * 70)
        print("[Brain Network Demo]")
        print("=" * 70)

        brain = BrainNetwork()
        sensory_input = torch.rand(1, 6)

        print("\nSensory input (random):")
        modalities = ['Visual', 'Auditory', 'Tactile', 'Olfactory', 'Gustatory', 'Proprioceptive']
        for mod, val in zip(modalities, sensory_input[0]):
            print(f"  {mod:15s}: {val.item():.3f}")

        outputs, state = brain(sensory_input)

        print("\nBrain state:")
        print(f"  Membrane potential: {outputs['membrane_potential'].item():.2f} mV")
        print(f"  Calcium: {outputs['calcium'].item():.2f} nM")
        print(f"  ATP: {outputs['ATP'].item():.2f} μM")
        print(f"  Neurotransmitter: {outputs['neurotransmitter'].item():.2f} μM")


def umi_demo():
    """Demonstrate UMI anomaly detection."""

    if STREAMLIT_AVAILABLE:
        st.header("🚨 UMI Anomaly Detection")

        st.markdown("""
        Unified Monitoring Index computes: **UMI = α·ΔR + β·T + γ·V + δ·A**

        Where:
        - ΔR: Relative deviation from baseline
        - T: Trend coefficient
        - V: Coefficient of variation
        - A: Anomaly score
        """)

        col1, col2 = st.columns(2)
        with col1:
            delta_r = st.slider("Deviation (ΔR)", -1.0, 1.0, 0.0)
            trend = st.slider("Trend (T)", -1.0, 1.0, 0.0)
        with col2:
            variation = st.slider("Variation (V)", 0.0, 1.0, 0.2)
            anomaly = st.slider("Anomaly (A)", -2.0, 2.0, 0.0)

        if st.button("📊 Compute UMI"):
            umi_net = UMI_Network()
            inputs = torch.tensor([[delta_r, trend, variation, anomaly]])

            umi_score, alert = umi_net(inputs, return_alerts=True)

            alert_level = alert.item()
            alert_names = ['🟢 NORMAL', '🟡 WARNING', '🔴 CRITICAL']
            alert_colors = ['green', 'orange', 'red']

            st.metric("UMI Score", f"{umi_score.item():.4f}")
            st.markdown(f"### Alert: :{alert_colors[alert_level]}[{alert_names[alert_level]}]")
    else:
        print("\n" + "=" * 70)
        print("[UMI Anomaly Detection Demo]")
        print("=" * 70)

        umi_net = UMI_Network()

        # Test cases
        test_cases = [
            ([0.1, 0.05, 0.2, -0.5], "Normal conditions"),
            ([0.5, 0.4, 0.3, 0.8], "Warning level"),
            ([1.0, 1.0, 1.0, 1.0], "Critical alert")
        ]

        for inputs, description in test_cases:
            tensor_input = torch.tensor([inputs])
            umi_score, alert = umi_net(tensor_input, return_alerts=True)

            alert_names = ['NORMAL', 'WARNING', 'CRITICAL']
            print(f"\n{description}:")
            print(f"  UMI: {umi_score.item():7.4f} → {alert_names[alert.item()]}")


def main():
    """Main demo function."""

    if STREAMLIT_AVAILABLE:
        # Streamlit web interface
        tab1, tab2, tab3 = st.tabs(["🧠 Unified Model", "🔬 Brain Network", "🚨 UMI Monitor"])

        with tab1:
            create_model_demo()

        with tab2:
            brain_demo()

        with tab3:
            umi_demo()

        # Footer
        st.markdown("---")
        st.markdown("""
        **NeuralLayers Framework** |
        [GitHub](https://github.com/biblicalandr0id/NeuralLayers) |
        [Documentation](../README.md)
        """)

    else:
        # Command-line interface
        print("\n" + "=" * 70)
        print(" " * 15 + "🧠 NEURALLAYERS INTERACTIVE DEMO")
        print("=" * 70)
        print("\nRunning all demonstrations...\n")

        create_model_demo()
        brain_demo()
        umi_demo()

        print("\n" + "=" * 70)
        print("✅ Demo complete!")
        print("\nTo run web interface: pip install streamlit && streamlit run demo_app.py")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
