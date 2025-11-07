"""
Unit tests for UnifiedBrainLogicNetwork and its components.
"""

import unittest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from logicalbrain_network import (
    UnifiedBrainLogicNetwork,
    CerebrumModule,
    CerebellumModule,
    BrainstemModule,
    SensoryProcessor,
    LogicalProcessor,
    NeuralLogicalIntegrator,
    SystemState,
    UnifiedOutput
)


class TestCerebrumModule(unittest.TestCase):
    """Test cases for CerebrumModule."""

    def setUp(self):
        self.hidden_dim = 64
        self.batch_size = 4
        self.module = CerebrumModule(self.hidden_dim)

    def test_initialization(self):
        """Test module initialization."""
        self.assertIsInstance(self.module, CerebrumModule)
        self.assertTrue(hasattr(self.module, 'executive_network'))
        self.assertTrue(hasattr(self.module, 'working_memory'))
        self.assertTrue(hasattr(self.module, 'reasoning_network'))

    def test_forward_pass(self):
        """Test forward pass with valid inputs."""
        sensory_input = torch.randn(self.batch_size, self.hidden_dim)
        logical_input = torch.randn(self.batch_size, self.hidden_dim)

        output = self.module(sensory_input, logical_input)

        self.assertEqual(output.shape, (self.batch_size, self.hidden_dim))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_output_range(self):
        """Test that outputs are in reasonable range."""
        sensory_input = torch.randn(self.batch_size, self.hidden_dim)
        logical_input = torch.randn(self.batch_size, self.hidden_dim)

        output = self.module(sensory_input, logical_input)

        # Check that outputs are not exploding
        self.assertTrue(output.abs().max() < 100.0)


class TestCerebellumModule(unittest.TestCase):
    """Test cases for CerebellumModule."""

    def setUp(self):
        self.hidden_dim = 64
        self.batch_size = 4
        self.module = CerebellumModule(self.hidden_dim)

    def test_initialization(self):
        """Test module initialization."""
        self.assertIsInstance(self.module, CerebellumModule)
        self.assertTrue(hasattr(self.module, 'motor_learning'))
        self.assertTrue(hasattr(self.module, 'error_correction'))
        self.assertTrue(hasattr(self.module, 'temporal_predictor'))

    def test_forward_pass(self):
        """Test forward pass with valid inputs."""
        sensory_input = torch.randn(self.batch_size, self.hidden_dim)
        truth_values = torch.randn(self.batch_size, self.hidden_dim)

        output = self.module(sensory_input, truth_values)

        self.assertEqual(output.shape, (self.batch_size, self.hidden_dim))
        self.assertFalse(torch.isnan(output).any())

    def test_temporal_state_persistence(self):
        """Test that temporal state is maintained across calls."""
        sensory_input = torch.randn(self.batch_size, self.hidden_dim)
        truth_values = torch.randn(self.batch_size, self.hidden_dim)

        output1 = self.module(sensory_input, truth_values)
        output2 = self.module(sensory_input, truth_values)

        # Outputs should be different due to hidden state
        self.assertFalse(torch.allclose(output1, output2))


class TestBrainstemModule(unittest.TestCase):
    """Test cases for BrainstemModule."""

    def setUp(self):
        self.hidden_dim = 64
        self.batch_size = 4
        self.module = BrainstemModule(self.hidden_dim)

    def test_initialization(self):
        """Test module initialization."""
        self.assertIsInstance(self.module, BrainstemModule)
        self.assertTrue(hasattr(self.module, 'arousal_system'))
        self.assertTrue(hasattr(self.module, 'autonomic_regulator'))
        self.assertTrue(hasattr(self.module, 'vital_monitor'))

    def test_forward_pass(self):
        """Test forward pass with valid inputs."""
        sensory_input = torch.randn(self.batch_size, self.hidden_dim)
        membrane_potential = torch.randn(self.batch_size, self.hidden_dim)

        output = self.module(sensory_input, membrane_potential)

        self.assertEqual(output.shape, (self.batch_size, self.hidden_dim))
        self.assertFalse(torch.isnan(output).any())

    def test_homeostatic_control(self):
        """Test homeostatic regulation."""
        sensory_input = torch.randn(self.batch_size, self.hidden_dim)
        # Extreme membrane potential
        membrane_potential = torch.full((self.batch_size, self.hidden_dim), 100.0)

        output = self.module(sensory_input, membrane_potential)

        # Output should be modulated by sigmoid
        self.assertTrue(output.abs().max() < 50.0)


class TestSystemState(unittest.TestCase):
    """Test cases for SystemState."""

    def setUp(self):
        self.hidden_dim = 64
        self.batch_size = 4
        self.system_state = SystemState(self.hidden_dim)

    def test_initialization(self):
        """Test state initialization."""
        x = torch.randn(self.batch_size, self.hidden_dim)
        state = self.system_state.initialize(x)

        # Check all components exist
        expected_components = ['V', 'NT', 'Ca', 'ATP', 'g', 'Ψ', 'τ', 'ω']
        for component in expected_components:
            self.assertIn(component, state)
            self.assertEqual(state[component].shape, (self.batch_size, self.hidden_dim))

    def test_update(self):
        """Test state update."""
        cerebrum_output = torch.randn(self.batch_size, self.hidden_dim)
        cerebellum_output = torch.randn(self.batch_size, self.hidden_dim)
        brainstem_output = torch.randn(self.batch_size, self.hidden_dim)

        updated_state = self.system_state.update(
            cerebrum_output,
            cerebellum_output,
            brainstem_output
        )

        # Check all components updated
        self.assertIn('V', updated_state)
        self.assertIn('ATP', updated_state)
        self.assertIn('Ca', updated_state)

        # Check biological constraints
        V = updated_state['V']
        ATP = updated_state['ATP']
        Ca = updated_state['Ca']

        self.assertTrue((V >= -70.0).all())
        self.assertTrue((V <= 40.0).all())
        self.assertTrue((ATP >= 1000.0).all())
        self.assertTrue((ATP <= 10000.0).all())
        self.assertTrue((Ca >= 0.0).all())
        self.assertTrue((Ca <= 1000.0).all())


class TestUnifiedBrainLogicNetwork(unittest.TestCase):
    """Test cases for UnifiedBrainLogicNetwork."""

    def setUp(self):
        self.input_dim = 128
        self.hidden_dim = 256
        self.output_dim = 64
        self.batch_size = 4
        self.network = UnifiedBrainLogicNetwork(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim
        )

    def test_initialization(self):
        """Test network initialization."""
        self.assertIsInstance(self.network, UnifiedBrainLogicNetwork)
        self.assertTrue(hasattr(self.network, 'cerebrum'))
        self.assertTrue(hasattr(self.network, 'cerebellum'))
        self.assertTrue(hasattr(self.network, 'brainstem'))

    def test_forward_pass(self):
        """Test complete forward pass."""
        x = torch.randn(self.batch_size, self.input_dim)

        output = self.network(x)

        # Check output dictionary
        self.assertIn('output', output)
        self.assertIn('system_state', output)
        self.assertIn('membrane_potential', output)
        self.assertIn('truth_values', output)

        # Check shapes
        self.assertEqual(output['output'].shape, (self.batch_size, self.output_dim))

    def test_constraints(self):
        """Test that constraints are applied."""
        x = torch.randn(self.batch_size, self.input_dim)

        output = self.network(x)

        # Check membrane potential constraints
        V = output['membrane_potential']
        self.assertTrue((V >= -70.0).all() or (V <= 40.0).all())

        # Check truth value constraints
        tau = output['truth_values']
        self.assertTrue((tau >= 0.0).all())
        self.assertTrue((tau <= 1.0).all())

    def test_gradient_flow(self):
        """Test that gradients flow through the network."""
        x = torch.randn(self.batch_size, self.input_dim, requires_grad=True)

        output = self.network(x)
        loss = output['output'].sum()
        loss.backward()

        # Check that input received gradients
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())

    def test_batch_size_flexibility(self):
        """Test that network handles different batch sizes."""
        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, self.input_dim)
            output = self.network(x)
            self.assertEqual(output['output'].shape[0], batch_size)


class TestSensoryProcessor(unittest.TestCase):
    """Test cases for SensoryProcessor."""

    def setUp(self):
        self.input_dim = 128
        self.hidden_dim = 256
        self.batch_size = 4
        self.processor = SensoryProcessor(self.input_dim, self.hidden_dim)

    def test_forward_pass(self):
        """Test sensory processing."""
        x = torch.randn(self.batch_size, self.input_dim)
        output = self.processor(x)

        self.assertEqual(output.shape, (self.batch_size, self.hidden_dim))
        self.assertFalse(torch.isnan(output).any())


class TestLogicalProcessor(unittest.TestCase):
    """Test cases for LogicalProcessor."""

    def setUp(self):
        self.hidden_dim = 256
        self.batch_size = 4
        self.processor = LogicalProcessor(self.hidden_dim)

    def test_forward_pass(self):
        """Test logical processing."""
        x = torch.randn(self.batch_size, self.hidden_dim)

        processed = self.processor.process_input(x)
        truth_values = self.processor.compute_truth_valuation(processed)

        self.assertEqual(truth_values.shape, (self.batch_size, self.hidden_dim))
        self.assertTrue((truth_values >= 0.0).all())
        self.assertTrue((truth_values <= 1.0).all())


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
