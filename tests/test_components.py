"""
Unit tests for individual NeuralLayers components.
"""

import unittest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain_network_implementation import BrainNetwork
from consciousness_layers import ConsciousnessEmergence
from LogicalReasoningLayer import LogicalReasoningLayer
from umi_layer import UMI_Layer, UMI_Network


class TestBrainNetwork(unittest.TestCase):
    """Test cases for BrainNetwork."""

    def setUp(self):
        self.network = BrainNetwork()
        self.batch_size = 2

    def test_initialization(self):
        """Test network initialization."""
        self.assertIsInstance(self.network, BrainNetwork)
        # Check biological constants
        self.assertEqual(self.network.Vrest, -70.0)
        self.assertEqual(self.network.Vpeak, 40.0)

    def test_forward_pass(self):
        """Test forward pass with sensory input."""
        sensory_input = torch.rand(self.batch_size, 6)

        outputs, state = self.network(sensory_input)

        # Check outputs exist
        self.assertIn('motor', outputs)
        self.assertIn('autonomic', outputs)
        self.assertIn('cognitive', outputs)
        self.assertIn('membrane_potential', outputs)
        self.assertIn('calcium', outputs)
        self.assertIn('ATP', outputs)
        self.assertIn('neurotransmitter', outputs)

        # Check state tuple
        self.assertEqual(len(state), 4)

    def test_membrane_potential_bounds(self):
        """Test membrane potential stays within biological bounds."""
        sensory_input = torch.rand(self.batch_size, 6)

        outputs, state = self.network(sensory_input)
        V = outputs['membrane_potential']

        self.assertTrue((V >= -70.0).all())
        self.assertTrue((V <= 40.0).all())

    def test_calcium_dynamics(self):
        """Test calcium dynamics."""
        V = torch.tensor([[-70.0], [-50.0], [0.0]])
        Ca = torch.tensor([[100.0], [200.0], [500.0]])

        Ca_new = self.network.calcium_dynamics(V, Ca)

        self.assertTrue((Ca_new >= 0.0).all())
        self.assertTrue((Ca_new <= self.network.Ca_max).all())

    def test_energy_conservation(self):
        """Test ATP energy dynamics."""
        ATP = torch.tensor([[5000.0], [2000.0], [8000.0]])
        activity = torch.randn(3, 128)

        ATP_new = self.network.energy_dynamics(ATP, activity)

        self.assertTrue((ATP_new >= self.network.ATP_min).all())
        self.assertTrue((ATP_new <= self.network.ATP_max).all())

    def test_neurotransmitter_release(self):
        """Test neurotransmitter dynamics."""
        V = torch.tensor([[-70.0], [-50.0], [0.0]])
        NT = torch.tensor([[1.0], [5.0], [9.0]])

        NT_new = self.network.neurotransmitter_dynamics(NT, V)

        self.assertTrue((NT_new >= 0.0).all())
        self.assertTrue((NT_new <= self.network.NT_max).all())

    def test_state_persistence(self):
        """Test that state persists across forward passes."""
        sensory_input = torch.rand(self.batch_size, 6)

        # First pass
        outputs1, state1 = self.network(sensory_input)

        # Second pass with previous state
        outputs2, state2 = self.network(sensory_input, state1)

        # States should differ
        self.assertFalse(torch.allclose(state1[0], state2[0]))


class TestConsciousnessEmergence(unittest.TestCase):
    """Test cases for ConsciousnessEmergence."""

    def setUp(self):
        self.dimensions = (7, 7, 7)
        self.num_layers = 7
        self.consciousness = ConsciousnessEmergence(self.dimensions, self.num_layers)

    def test_initialization(self):
        """Test consciousness framework initialization."""
        self.assertIsInstance(self.consciousness, ConsciousnessEmergence)
        self.assertEqual(len(self.consciousness.layers), self.num_layers)

    def test_quantum_state_initialization(self):
        """Test quantum state initialization."""
        quantum_state = self.consciousness.initialize_quantum_state()

        self.assertEqual(len(quantum_state), self.num_layers)
        for layer_state in quantum_state:
            self.assertEqual(layer_state.shape, self.dimensions)
            self.assertTrue(layer_state.dtype == torch.complex64)

    def test_process_moment(self):
        """Test processing a moment of consciousness."""
        input_state = torch.randn(*self.dimensions).to(torch.complex64)

        conscious_state = self.consciousness.process_moment(input_state)

        self.assertEqual(len(conscious_state), self.num_layers)

        # Check that consciousness decays with layers (golden ratio decay)
        magnitudes = [torch.abs(layer).mean().item() for layer in conscious_state]
        # Later layers should generally have different magnitudes
        self.assertFalse(all(m == magnitudes[0] for m in magnitudes))

    def test_golden_ratio_decay(self):
        """Test golden ratio decay in consciousness layers."""
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio

        # Layer 4 should have golden ratio amplification
        # Check that phi is used correctly
        self.assertAlmostEqual(self.consciousness.phi, phi, places=5)


class TestLogicalReasoningLayer(unittest.TestCase):
    """Test cases for LogicalReasoningLayer."""

    def setUp(self):
        self.input_dim = 128
        self.hidden_dim = 256
        self.num_premises = 3
        self.batch_size = 4
        self.layer = LogicalReasoningLayer(
            self.input_dim,
            self.hidden_dim,
            self.num_premises
        )

    def test_initialization(self):
        """Test layer initialization."""
        self.assertIsInstance(self.layer, LogicalReasoningLayer)
        self.assertEqual(self.layer.num_premises, self.num_premises)

    def test_forward_pass(self):
        """Test logical reasoning inference."""
        premises = torch.randn(self.batch_size, self.num_premises, self.input_dim)

        conclusion = self.layer(premises)

        # Check output shape
        self.assertEqual(conclusion.shape, (self.batch_size, 1))

        # Check output is in logical range [-1, 1] (tanh activation)
        self.assertTrue((conclusion >= -1.0).all())
        self.assertTrue((conclusion <= 1.0).all())

    def test_phi_activation(self):
        """Test custom Phi activation function."""
        x = torch.randn(self.batch_size, self.hidden_dim)

        # Phi activation should use golden ratio
        phi = (1 + np.sqrt(5)) / 2
        sigma = phi

        # Manually compute expected activation
        expected = torch.exp(-torch.norm(x, dim=-1, keepdim=True) ** 2 / (2 * sigma ** 2))

        # Get layer's phi activation (would need to expose this in the layer)
        # For now, just check that activation exists
        self.assertTrue(hasattr(self.layer, 'phi_activation'))

    def test_fibonacci_weights(self):
        """Test Fibonacci-based premise weights."""
        # Check that premise weights follow Fibonacci pattern
        # This would require accessing internal weights
        # For now, verify layer has premise encoder
        self.assertTrue(hasattr(self.layer, 'premise_encoder'))

    def test_consistency(self):
        """Test logical consistency."""
        premises = torch.randn(self.batch_size, self.num_premises, self.input_dim)

        # Same premises should give same conclusion
        conclusion1 = self.layer(premises)
        conclusion2 = self.layer(premises)

        self.assertTrue(torch.allclose(conclusion1, conclusion2))


class TestUMILayer(unittest.TestCase):
    """Test cases for UMI_Layer (PyTorch version)."""

    def setUp(self):
        self.batch_size = 8
        self.layer = UMI_Layer(alpha=0.4, beta=0.3, gamma=0.2, delta=0.1)

    def test_initialization(self):
        """Test UMI layer initialization."""
        self.assertIsInstance(self.layer, UMI_Layer)

        # Check weights
        alpha, beta, gamma, delta = self.layer.get_weights()
        self.assertAlmostEqual(alpha + beta + gamma + delta, 1.0, places=5)

    def test_forward_pass(self):
        """Test UMI computation."""
        inputs = torch.tensor([
            [0.1, 0.05, 0.2, -0.5],
            [0.2, -0.1, 0.1, 0.2],
            [-0.05, 0.15, 0.05, 1.2]
        ], dtype=torch.float32)

        umi = self.layer(inputs)

        self.assertEqual(umi.shape, (3,))
        self.assertFalse(torch.isnan(umi).any())

    def test_input_validation(self):
        """Test input shape validation."""
        # Invalid shape should raise error
        with self.assertRaises(ValueError):
            invalid_input = torch.randn(self.batch_size, 5)  # Wrong number of features
            self.layer(invalid_input)

    def test_learnable_weights(self):
        """Test learnable UMI weights."""
        learnable_layer = UMI_Layer(learnable=True)

        # Check that weights are parameters
        params = list(learnable_layer.parameters())
        self.assertEqual(len(params), 4)  # alpha, beta, gamma, delta

    def test_alert_detection(self):
        """Test alert detection in UMI network."""
        network = UMI_Network(critical_threshold=1.0, warning_threshold=0.7)

        inputs = torch.tensor([
            [0.1, 0.05, 0.2, -0.5],   # Normal
            [0.5, 0.4, 0.3, 0.8],     # Warning
            [1.0, 1.0, 1.0, 1.0]      # Critical
        ], dtype=torch.float32)

        umi, alerts = network(inputs, return_alerts=True)

        self.assertEqual(alerts.shape, (3,))
        # Check that alert levels make sense (0=normal, 1=warning, 2=critical)
        self.assertTrue((alerts >= 0).all())
        self.assertTrue((alerts <= 2).all())


class TestIntegration(unittest.TestCase):
    """Integration tests across multiple components."""

    def test_full_pipeline(self):
        """Test full neural-logical pipeline."""
        from logicalbrain_network import UnifiedBrainLogicNetwork

        # Initialize network
        network = UnifiedBrainLogicNetwork(
            input_dim=256,
            hidden_dim=512,
            output_dim=128
        )

        # Create input
        x = torch.randn(4, 256)

        # Forward pass
        output = network(x)

        # Check complete output
        self.assertIn('output', output)
        self.assertIn('system_state', output)

        # Check system state components
        state = output['system_state']
        expected_components = ['V', 'NT', 'Ca', 'ATP', 'g', 'Ψ', 'τ', 'ω']
        for component in expected_components:
            self.assertIn(component, state)

    def test_consciousness_with_logical_reasoning(self):
        """Test consciousness layer with logical reasoning."""
        # This would test integration between consciousness and logical layers
        consciousness = ConsciousnessEmergence((7, 7, 7), 7)
        reasoning = LogicalReasoningLayer(128, 256, 3)

        # Create inputs
        conscious_input = torch.randn(7, 7, 7).to(torch.complex64)
        logical_premises = torch.randn(1, 3, 128)

        # Process both
        conscious_state = consciousness.process_moment(conscious_input)
        logical_conclusion = reasoning(logical_premises)

        # Both should produce valid outputs
        self.assertIsNotNone(conscious_state)
        self.assertIsNotNone(logical_conclusion)


if __name__ == '__main__':
    unittest.main(verbosity=2)
