#!/usr/bin/env python
"""
NeuralLayers Health Check & Verification Script

Verifies that all core components are working correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import traceback
from typing import Dict, Tuple, List


class HealthCheck:
    """Comprehensive health check for NeuralLayers"""

    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0

    def test(self, name: str, func):
        """Run a single test"""
        try:
            func()
            self.results.append((name, "✅ PASS", None))
            self.passed += 1
            print(f"✅ {name}")
        except Exception as e:
            self.results.append((name, "❌ FAIL", str(e)))
            self.failed += 1
            print(f"❌ {name}")
            print(f"   Error: {str(e)[:100]}")

    def test_imports(self):
        """Test that all modules can be imported"""
        print("\n" + "="*70)
        print("Testing Imports")
        print("="*70)

        def test_torch():
            import torch
            assert torch.cuda.is_available() or True  # CPU is okay

        def test_numpy():
            import numpy as np

        def test_yaml():
            import yaml

        def test_matplotlib():
            import matplotlib.pyplot as plt

        def test_logicalbrain():
            from logicalbrain_network import UnifiedBrainLogicNetwork

        def test_brain_network():
            from brain_network_implementation import BrainNetwork

        def test_logical_reasoning():
            from LogicalReasoningLayer import LogicalReasoningLayer

        def test_consciousness():
            from consciousness_layers import ConsciousnessEmergence

        def test_umi():
            from umi_layer import UMI_Layer

        self.test("Import torch", test_torch)
        self.test("Import numpy", test_numpy)
        self.test("Import yaml", test_yaml)
        self.test("Import matplotlib", test_matplotlib)
        self.test("Import UnifiedBrainLogicNetwork", test_logicalbrain)
        self.test("Import BrainNetwork", test_brain_network)
        self.test("Import LogicalReasoningLayer", test_logical_reasoning)
        self.test("Import ConsciousnessEmergence", test_consciousness)
        self.test("Import UMI_Layer", test_umi)

    def test_models(self):
        """Test that models can be instantiated"""
        print("\n" + "="*70)
        print("Testing Model Instantiation")
        print("="*70)

        def test_unified_brain():
            from logicalbrain_network import UnifiedBrainLogicNetwork
            model = UnifiedBrainLogicNetwork(
                input_dim=64,
                hidden_dim=128,
                output_dim=32
            )
            assert model is not None

        def test_brain_network():
            from brain_network_implementation import BrainNetwork
            model = BrainNetwork()
            assert model is not None

        def test_logical_layer():
            from LogicalReasoningLayer import LogicalReasoningLayer
            layer = LogicalReasoningLayer(
                input_dim=64,
                hidden_dim=128,
                num_premises=3
            )
            assert layer is not None

        def test_consciousness_layer():
            from consciousness_layers import ConsciousnessEmergence
            consciousness = ConsciousnessEmergence(
                dimensions=(7, 7, 7),
                layers=7
            )
            assert consciousness is not None

        def test_umi_layer():
            from umi_layer import UMI_Layer
            umi = UMI_Layer()
            assert umi is not None

        self.test("Instantiate UnifiedBrainLogicNetwork", test_unified_brain)
        self.test("Instantiate BrainNetwork", test_brain_network)
        self.test("Instantiate LogicalReasoningLayer", test_logical_layer)
        self.test("Instantiate ConsciousnessEmergence", test_consciousness_layer)
        self.test("Instantiate UMI_Layer", test_umi_layer)

    def test_forward_passes(self):
        """Test that models can perform forward passes"""
        print("\n" + "="*70)
        print("Testing Forward Passes")
        print("="*70)

        def test_unified_forward():
            from logicalbrain_network import UnifiedBrainLogicNetwork
            model = UnifiedBrainLogicNetwork(
                input_dim=64,
                hidden_dim=128,
                output_dim=32
            )
            x = torch.randn(4, 64)
            output = model(x)
            assert 'output' in output
            assert output['output'].shape == (4, 32)

        def test_brain_forward():
            from brain_network_implementation import BrainNetwork
            model = BrainNetwork()
            x = torch.randn(1, 6)  # 6 sensory modalities
            outputs, state = model(x)
            assert 'motor' in outputs
            assert 'cognitive' in outputs
            assert 'autonomic' in outputs

        def test_logical_forward():
            from LogicalReasoningLayer import LogicalReasoningLayer
            layer = LogicalReasoningLayer(
                input_dim=64,
                hidden_dim=128,
                num_premises=3
            )
            premises = torch.randn(2, 3, 64)
            output = layer(premises)
            assert output.shape[0] == 2

        def test_consciousness_forward():
            from consciousness_layers import ConsciousnessEmergence
            consciousness = ConsciousnessEmergence(
                dimensions=(7, 7, 7),
                layers=7
            )
            x = torch.randn(7, 7, 7).to(torch.complex64)
            output = consciousness.process_moment(x)
            assert len(output) == 7

        def test_umi_forward():
            from umi_layer import UMI_Layer
            umi = UMI_Layer()
            x = torch.randn(10, 4)  # 4 metrics
            output = umi(x)
            assert output.shape == (10,)

        self.test("Forward pass - UnifiedBrainLogicNetwork", test_unified_forward)
        self.test("Forward pass - BrainNetwork", test_brain_forward)
        self.test("Forward pass - LogicalReasoningLayer", test_logical_forward)
        self.test("Forward pass - ConsciousnessEmergence", test_consciousness_forward)
        self.test("Forward pass - UMI_Layer", test_umi_forward)

    def test_biological_constraints(self):
        """Test that biological constraints are satisfied"""
        print("\n" + "="*70)
        print("Testing Biological Constraints")
        print("="*70)

        def test_membrane_potential():
            from logicalbrain_network import UnifiedBrainLogicNetwork
            model = UnifiedBrainLogicNetwork(
                input_dim=64,
                hidden_dim=128,
                output_dim=32
            )
            x = torch.randn(4, 64)
            output = model(x)
            V = output['system_state']['V']
            assert (V >= -70.0).all() and (V <= 40.0).all(), f"V out of range: [{V.min():.2f}, {V.max():.2f}]"

        def test_neurotransmitters():
            from logicalbrain_network import UnifiedBrainLogicNetwork
            model = UnifiedBrainLogicNetwork(
                input_dim=64,
                hidden_dim=128,
                output_dim=32
            )
            x = torch.randn(4, 64)
            output = model(x)
            NT = output['system_state']['NT']
            assert (NT >= 0.0).all() and (NT <= 10.0).all(), f"NT out of range: [{NT.min():.2f}, {NT.max():.2f}]"

        def test_calcium():
            from logicalbrain_network import UnifiedBrainLogicNetwork
            model = UnifiedBrainLogicNetwork(
                input_dim=64,
                hidden_dim=128,
                output_dim=32
            )
            x = torch.randn(4, 64)
            output = model(x)
            Ca = output['system_state']['Ca']
            assert (Ca >= 0.0).all() and (Ca <= 1000.0).all(), f"Ca out of range: [{Ca.min():.2f}, {Ca.max():.2f}]"

        def test_atp():
            from logicalbrain_network import UnifiedBrainLogicNetwork
            model = UnifiedBrainLogicNetwork(
                input_dim=64,
                hidden_dim=128,
                output_dim=32
            )
            x = torch.randn(4, 64)
            output = model(x)
            ATP = output['system_state']['ATP']
            assert (ATP >= 1000.0).all() and (ATP <= 10000.0).all(), f"ATP out of range: [{ATP.min():.2f}, {ATP.max():.2f}]"

        def test_glial_state():
            from logicalbrain_network import UnifiedBrainLogicNetwork
            model = UnifiedBrainLogicNetwork(
                input_dim=64,
                hidden_dim=128,
                output_dim=32
            )
            x = torch.randn(4, 64)
            output = model(x)
            g = output['system_state']['g']
            assert (g >= 0.0).all() and (g <= 1.0).all(), f"g out of range: [{g.min():.4f}, {g.max():.4f}]"

        def test_truth_values():
            from logicalbrain_network import UnifiedBrainLogicNetwork
            model = UnifiedBrainLogicNetwork(
                input_dim=64,
                hidden_dim=128,
                output_dim=32
            )
            x = torch.randn(4, 64)
            output = model(x)
            tau = output['system_state']['tau']
            assert (tau >= 0.0).all() and (tau <= 1.0).all(), f"tau out of range: [{tau.min():.4f}, {tau.max():.4f}]"

        self.test("Membrane potential constraints (-70 to +40 mV)", test_membrane_potential)
        self.test("Neurotransmitter constraints (0 to 10 μM)", test_neurotransmitters)
        self.test("Calcium constraints (0 to 1000 nM)", test_calcium)
        self.test("ATP constraints (1000 to 10000 μM)", test_atp)
        self.test("Glial state constraints (0 to 1)", test_glial_state)
        self.test("Truth value constraints (0 to 1)", test_truth_values)

    def test_gradient_flow(self):
        """Test that gradients can flow through models"""
        print("\n" + "="*70)
        print("Testing Gradient Flow")
        print("="*70)

        def test_unified_gradients():
            from logicalbrain_network import UnifiedBrainLogicNetwork
            model = UnifiedBrainLogicNetwork(
                input_dim=64,
                hidden_dim=128,
                output_dim=32
            )
            x = torch.randn(4, 64, requires_grad=True)
            output = model(x)
            loss = output['output'].sum()
            loss.backward()
            assert x.grad is not None
            assert not torch.isnan(x.grad).any()

        def test_brain_gradients():
            from brain_network_implementation import BrainNetwork
            model = BrainNetwork()
            x = torch.randn(1, 6, requires_grad=True)
            outputs, state = model(x)
            loss = outputs['motor'].sum()
            loss.backward()
            assert x.grad is not None
            assert not torch.isnan(x.grad).any()

        self.test("Gradient flow - UnifiedBrainLogicNetwork", test_unified_gradients)
        self.test("Gradient flow - BrainNetwork", test_brain_gradients)

    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*70)
        print("HEALTH CHECK SUMMARY")
        print("="*70)

        print(f"\nTests Run:    {self.passed + self.failed}")
        print(f"Passed:       {self.passed} ✅")
        print(f"Failed:       {self.failed} ❌")
        print(f"Success Rate: {100 * self.passed / (self.passed + self.failed):.1f}%")

        if self.failed > 0:
            print("\n" + "="*70)
            print("FAILED TESTS")
            print("="*70)
            for name, status, error in self.results:
                if status == "❌ FAIL":
                    print(f"\n{name}:")
                    print(f"  {error}")

        print("\n" + "="*70)
        if self.failed == 0:
            print("✅ ALL TESTS PASSED - System is healthy!")
        else:
            print(f"⚠️  {self.failed} test(s) failed - Review errors above")
        print("="*70)

        return self.failed == 0

    def run_all(self):
        """Run all health checks"""
        print("🏥 NeuralLayers Health Check")
        print("="*70)
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available:  {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Version:    {torch.version.cuda}")
            print(f"GPU:             {torch.cuda.get_device_name(0)}")
        print("="*70)

        self.test_imports()
        self.test_models()
        self.test_forward_passes()
        self.test_biological_constraints()
        self.test_gradient_flow()

        return self.print_summary()


def main():
    """Run health check"""
    health_check = HealthCheck()
    success = health_check.run_all()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
