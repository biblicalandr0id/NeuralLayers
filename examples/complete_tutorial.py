"""
Complete NeuralLayers Tutorial

This tutorial demonstrates all major features of the NeuralLayers framework:
1. Configuration management
2. Model initialization and training
3. State tracking and visualization
4. Checkpointing
5. Logical reasoning
6. Consciousness modeling
7. UMI monitoring

Author: biblicalandr0id
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Import NeuralLayers components
from logicalbrain_network import UnifiedBrainLogicNetwork
from brain_network_implementation import BrainNetwork
from consciousness_layers import ConsciousnessEmergence
from LogicalReasoningLayer import LogicalReasoningLayer
from umi_layer import UMI_Network
from utils import (
    Config, Logger, CheckpointManager, StateVisualizer,
    Profiler, DeviceManager, InputValidator, GradientClipper
)


def main():
    print("=" * 80)
    print(" " * 20 + "NEURALLAYERS COMPLETE TUTORIAL")
    print("=" * 80)

    # =========================================================================
    # 1. Configuration
    # =========================================================================
    print("\n[Step 1] Loading Configuration")
    print("-" * 80)

    config_path = Path(__file__).parent.parent / "config.yaml"
    config = Config(str(config_path) if config_path.exists() else None)

    print(f"✓ Model dimensions: {config.get('model.input_dim')} → "
          f"{config.get('model.hidden_dim')} → {config.get('model.output_dim')}")
    print(f"✓ Batch size: {config.get('training.batch_size')}")
    print(f"✓ Learning rate: {config.get('training.learning_rate')}")

    # =========================================================================
    # 2. Device Setup
    # =========================================================================
    print("\n[Step 2] Setting Up Device")
    print("-" * 80)

    device_manager = DeviceManager(config)
    device = device_manager.device

    # =========================================================================
    # 3. Logger Initialization
    # =========================================================================
    print("\n[Step 3] Initializing Logger")
    print("-" * 80)

    logger = Logger("NeuralLayers Tutorial", config)
    logger.info("Tutorial started")

    # =========================================================================
    # 4. Model Initialization
    # =========================================================================
    print("\n[Step 4] Initializing Models")
    print("-" * 80)

    # Main unified network
    model = UnifiedBrainLogicNetwork(
        input_dim=config.get('model.input_dim'),
        hidden_dim=config.get('model.hidden_dim'),
        output_dim=config.get('model.output_dim')
    ).to(device)

    logger.info(f"Unified Network initialized with "
                f"{sum(p.numel() for p in model.parameters())} parameters")

    # Brain network
    brain = BrainNetwork().to(device)
    logger.info("Brain Network initialized")

    # Consciousness
    consciousness = ConsciousnessEmergence((7, 7, 7), 7)
    logger.info("Consciousness framework initialized")

    # Logical reasoning
    reasoning = LogicalReasoningLayer(128, 256, 3).to(device)
    logger.info("Logical Reasoning layer initialized")

    # UMI monitoring
    umi_network = UMI_Network().to(device)
    logger.info("UMI Monitoring network initialized")

    # =========================================================================
    # 5. Optimizer and Loss
    # =========================================================================
    print("\n[Step 5] Setting Up Optimization")
    print("-" * 80)

    optimizer = optim.Adam(model.parameters(),
                          lr=config.get('training.learning_rate'))
    criterion = nn.MSELoss()
    gradient_clipper = GradientClipper(
        clip_norm=config.get('numerical.gradient_clip_norm')
    )

    logger.info("Optimizer: Adam")
    logger.info("Loss: MSE")

    # =========================================================================
    # 6. Checkpoint Manager
    # =========================================================================
    print("\n[Step 6] Checkpoint Manager")
    print("-" * 80)

    checkpoint_mgr = CheckpointManager(
        save_dir=config.get('checkpointing.save_dir', './checkpoints'),
        keep_last_n=config.get('checkpointing.keep_last_n', 5)
    )

    logger.info(f"Checkpoints will be saved to: {checkpoint_mgr.save_dir}")

    # =========================================================================
    # 7. Visualization Setup
    # =========================================================================
    print("\n[Step 7] Visualization Setup")
    print("-" * 80)

    visualizer = StateVisualizer(
        save_dir=config.get('visualization.plot_dir', './plots')
    )

    logger.info(f"Plots will be saved to: {visualizer.save_dir}")

    # =========================================================================
    # 8. Training Loop (Demo)
    # =========================================================================
    print("\n[Step 8] Training Demonstration")
    print("-" * 80)

    profiler = Profiler()
    num_steps = 10  # Demo with 10 steps
    batch_size = config.get('training.batch_size')
    input_dim = config.get('model.input_dim')

    training_metrics = {'loss': [], 'atp_level': [], 'truth_accuracy': []}

    logger.info("Starting training loop...")

    for step in range(num_steps):
        # Generate random training data
        profiler.start("data_generation")
        x = torch.randn(batch_size, input_dim).to(device)
        target = torch.randn(batch_size, config.get('model.output_dim')).to(device)
        profiler.end("data_generation")

        # Validate input
        profiler.start("input_validation")
        validator = InputValidator()
        validator.validate_shape(x, (batch_size, input_dim), "input")
        validator.check_nan_inf(x, "input")
        profiler.end("input_validation")

        # Forward pass
        profiler.start("forward_pass")
        optimizer.zero_grad()
        output = model(x)
        profiler.end("forward_pass")

        # Compute loss
        profiler.start("loss_computation")
        loss = criterion(output['output'], target)
        profiler.end("loss_computation")

        # Backward pass
        profiler.start("backward_pass")
        loss.backward()
        gradient_clipper.clip(model)
        optimizer.step()
        profiler.end("backward_pass")

        # Track metrics
        training_metrics['loss'].append(loss.item())
        training_metrics['atp_level'].append(
            output['system_state']['ATP'].mean().item()
        )
        training_metrics['truth_accuracy'].append(
            output['truth_values'].mean().item()
        )

        # Log progress
        if (step + 1) % 5 == 0:
            logger.info(f"Step {step + 1}/{num_steps} | Loss: {loss.item():.4f} | "
                       f"ATP: {training_metrics['atp_level'][-1]:.2f}")

        # Save checkpoint
        if (step + 1) % 5 == 0:
            checkpoint_mgr.save(
                model=model,
                optimizer=optimizer,
                step=step,
                metrics={'loss': loss.item()}
            )
            logger.info(f"Checkpoint saved at step {step + 1}")

    # =========================================================================
    # 9. Consciousness Processing
    # =========================================================================
    print("\n[Step 9] Consciousness Processing")
    print("-" * 80)

    logger.info("Processing consciousness layers...")

    # Create initial quantum state
    quantum_state = consciousness.initialize_quantum_state()

    # Process a moment
    input_moment = torch.randn(7, 7, 7).to(torch.complex64)
    conscious_output = consciousness.process_moment(input_moment)

    logger.info(f"Consciousness processed across {len(conscious_output)} layers")
    for i, layer_state in enumerate(conscious_output):
        magnitude = torch.abs(layer_state).mean().item()
        logger.info(f"  Layer {i}: Magnitude = {magnitude:.4f}")

    # =========================================================================
    # 10. Logical Reasoning Demo
    # =========================================================================
    print("\n[Step 10] Logical Reasoning")
    print("-" * 80)

    logger.info("Performing logical reasoning...")

    # Create premises
    premises = torch.randn(1, 3, 128).to(device)

    # Reason
    conclusion = reasoning(premises)

    logger.info(f"Logical conclusion (truth value): {conclusion.item():.4f}")

    # =========================================================================
    # 11. Brain Network Simulation
    # =========================================================================
    print("\n[Step 11] Brain Network Simulation")
    print("-" * 80)

    logger.info("Simulating biological brain network...")

    # Multi-modal sensory input
    sensory_input = torch.rand(1, 6).to(device)  # 6 sensory modalities

    # Process
    brain_outputs, brain_state = brain(sensory_input)

    logger.info("Brain state:")
    logger.info(f"  Membrane Potential: {brain_outputs['membrane_potential'].item():.2f} mV")
    logger.info(f"  Calcium: {brain_outputs['calcium'].item():.2f} nM")
    logger.info(f"  ATP: {brain_outputs['ATP'].item():.2f} μM")
    logger.info(f"  Neurotransmitter: {brain_outputs['neurotransmitter'].item():.2f} μM")

    # =========================================================================
    # 12. UMI Monitoring
    # =========================================================================
    print("\n[Step 12] UMI Anomaly Detection")
    print("-" * 80)

    logger.info("Running UMI monitoring...")

    # Monitoring metrics [DeltaR, T, V, A]
    monitoring_data = torch.tensor([
        [0.1, 0.05, 0.2, -0.5],  # Normal
        [0.5, 0.4, 0.3, 0.8],    # Warning
        [1.0, 1.0, 1.0, 1.0]     # Critical
    ]).to(device)

    umi_scores, alerts = umi_network(monitoring_data, return_alerts=True)

    alert_names = ['NORMAL', 'WARNING', 'CRITICAL']
    for i, (score, alert) in enumerate(zip(umi_scores, alerts)):
        logger.info(f"  Sample {i+1}: UMI={score.item():.4f} -> {alert_names[alert.item()]}")

    # =========================================================================
    # 13. System State Visualization
    # =========================================================================
    print("\n[Step 13] Visualization")
    print("-" * 80)

    logger.info("Generating visualizations...")

    # Visualize system state
    final_output = model(x)
    visualizer.plot_system_state(final_output['system_state'], step=num_steps)
    logger.info(f"System state plot saved")

    # Plot training metrics
    visualizer.plot_training_metrics(training_metrics)
    logger.info(f"Training metrics plot saved")

    # =========================================================================
    # 14. Performance Profiling
    # =========================================================================
    print("\n[Step 14] Performance Report")
    print("-" * 80)

    print(profiler.report())

    # =========================================================================
    # 15. Model Export
    # =========================================================================
    print("\n[Step 15] Model Export")
    print("-" * 80)

    logger.info("Exporting final model...")

    # Save final checkpoint
    final_checkpoint = checkpoint_mgr.save(
        model=model,
        optimizer=optimizer,
        step=num_steps,
        metrics=training_metrics,
        config=config.config
    )

    logger.info(f"Final model saved to: {final_checkpoint}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print(" " * 30 + "TUTORIAL COMPLETE!")
    print("=" * 80)

    print("\n📊 Summary:")
    print(f"  • Training steps: {num_steps}")
    print(f"  • Final loss: {training_metrics['loss'][-1]:.4f}")
    print(f"  • Average ATP: {sum(training_metrics['atp_level'])/len(training_metrics['atp_level']):.2f} μM")
    print(f"  • Truth accuracy: {training_metrics['truth_accuracy'][-1]:.4f}")
    print(f"  • Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  • Device: {device}")

    print("\n📁 Outputs:")
    print(f"  • Checkpoints: {checkpoint_mgr.save_dir}")
    print(f"  • Plots: {visualizer.save_dir}")
    print(f"  • Logs: neurallayers.log")

    print("\n✅ All systems operational!")
    print("=" * 80)

    logger.info("Tutorial completed successfully")


if __name__ == "__main__":
    # Create examples directory if it doesn't exist
    os.makedirs(os.path.dirname(__file__), exist_ok=True)

    # Run tutorial
    main()
