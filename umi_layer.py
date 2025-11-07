"""
UMI Layer (Unified Monitoring Index) - PyTorch Implementation

Migrated from TensorFlow to PyTorch for framework unification.

The UMI layer computes a weighted anomaly detection score based on:
- DeltaR: Relative deviation from baseline
- T: Trend coefficient
- V: Coefficient of variation
- A: Anomaly score

Formula: UMI = α·ΔR + β·T + γ·V + δ·A

Default weights: α=0.4, β=0.3, γ=0.2, δ=0.1
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class UMI_Layer(nn.Module):
    """
    Unified Monitoring Index (UMI) Layer for anomaly detection.

    Computes a weighted combination of monitoring metrics to produce
    a unified anomaly score.

    Args:
        alpha (float): Weight for relative deviation (default: 0.4)
        beta (float): Weight for trend coefficient (default: 0.3)
        gamma (float): Weight for coefficient of variation (default: 0.2)
        delta (float): Weight for anomaly score (default: 0.1)
        learnable (bool): If True, weights are learnable parameters (default: False)

    Input Shape:
        (batch_size, 4): [DeltaR, T, V, A]

    Output Shape:
        (batch_size,): UMI scores
    """

    def __init__(
        self,
        alpha: float = 0.4,
        beta: float = 0.3,
        gamma: float = 0.2,
        delta: float = 0.1,
        learnable: bool = False
    ):
        super(UMI_Layer, self).__init__()

        # Validate weights sum to 1.0
        total = alpha + beta + gamma + delta
        if not (0.99 <= total <= 1.01):
            print(f"Warning: Weights sum to {total:.3f}, normalizing to 1.0")
            alpha, beta, gamma, delta = (
                alpha / total,
                beta / total,
                gamma / total,
                delta / total
            )

        if learnable:
            # Learnable weights (can be optimized during training)
            self.alpha = nn.Parameter(torch.tensor(alpha))
            self.beta = nn.Parameter(torch.tensor(beta))
            self.gamma = nn.Parameter(torch.tensor(gamma))
            self.delta = nn.Parameter(torch.tensor(delta))
        else:
            # Fixed weights (registered as buffers, not parameters)
            self.register_buffer('alpha', torch.tensor(alpha))
            self.register_buffer('beta', torch.tensor(beta))
            self.register_buffer('gamma', torch.tensor(gamma))
            self.register_buffer('delta', torch.tensor(delta))

        self.learnable = learnable

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute UMI scores.

        Args:
            inputs: Tensor of shape (batch_size, 4) containing [DeltaR, T, V, A]

        Returns:
            UMI scores of shape (batch_size,)

        Raises:
            ValueError: If input shape is incorrect
        """
        # Validate input shape
        if inputs.dim() != 2 or inputs.size(1) != 4:
            raise ValueError(
                f"Input must have shape (batch_size, 4), got {inputs.shape}"
            )

        # Extract components
        delta_r = inputs[:, 0]  # Relative deviation from baseline
        t = inputs[:, 1]        # Trend coefficient
        v = inputs[:, 2]        # Coefficient of variation
        a = inputs[:, 3]        # Anomaly score

        # Compute weighted UMI
        umi = (
            self.alpha * delta_r +
            self.beta * t +
            self.gamma * v +
            self.delta * a
        )

        return umi

    def get_weights(self) -> Tuple[float, float, float, float]:
        """
        Get current weight values.

        Returns:
            Tuple of (alpha, beta, gamma, delta)
        """
        return (
            self.alpha.item(),
            self.beta.item(),
            self.gamma.item(),
            self.delta.item()
        )

    def extra_repr(self) -> str:
        """String representation for print/summary."""
        return (
            f"alpha={self.alpha.item():.3f}, "
            f"beta={self.beta.item():.3f}, "
            f"gamma={self.gamma.item():.3f}, "
            f"delta={self.delta.item():.3f}, "
            f"learnable={self.learnable}"
        )


class UMI_Network(nn.Module):
    """
    Complete UMI monitoring network with preprocessing and alert detection.

    Args:
        hidden_dim (int): Hidden layer dimension (default: 64)
        critical_threshold (float): Threshold for critical alerts (default: 1.0)
        warning_threshold (float): Threshold for warnings (default: 0.7)
        learnable_weights (bool): Use learnable UMI weights (default: False)
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        critical_threshold: float = 1.0,
        warning_threshold: float = 0.7,
        learnable_weights: bool = False
    ):
        super(UMI_Network, self).__init__()

        self.critical_threshold = critical_threshold
        self.warning_threshold = warning_threshold

        # Preprocessing layer (optional feature extraction)
        self.preprocessing = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 4),
            nn.Tanh()  # Normalize to [-1, 1]
        )

        # UMI computation layer
        self.umi_layer = UMI_Layer(learnable=learnable_weights)

    def forward(
        self,
        inputs: torch.Tensor,
        return_alerts: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute UMI scores with optional alert detection.

        Args:
            inputs: Raw monitoring metrics (batch_size, 4)
            return_alerts: If True, also return alert tensor

        Returns:
            umi_scores: UMI values (batch_size,)
            alerts: Optional alert levels (batch_size,) if return_alerts=True
                   0 = normal, 1 = warning, 2 = critical
        """
        # Preprocess inputs
        processed = self.preprocessing(inputs)

        # Compute UMI
        umi = self.umi_layer(processed)

        if return_alerts:
            # Generate alert levels
            alerts = torch.zeros_like(umi, dtype=torch.long)
            alerts[torch.abs(umi) > self.warning_threshold] = 1  # Warning
            alerts[torch.abs(umi) > self.critical_threshold] = 2  # Critical
            return umi, alerts

        return umi, None


# Example Usage
if __name__ == "__main__":
    print("=" * 60)
    print("UMI Layer - PyTorch Implementation")
    print("=" * 60)

    # Example 1: Basic UMI Layer
    print("\n[Example 1] Basic UMI Layer")
    print("-" * 60)

    umi_layer = UMI_Layer()
    print(f"UMI Layer: {umi_layer}")

    # Example input data
    example_data = torch.tensor([
        [0.1, 0.05, 0.2, -0.5],  # Normal conditions
        [0.2, -0.1, 0.1, 0.2],   # Slight deviation
        [-0.05, 0.15, 0.05, 1.2] # High anomaly
    ], dtype=torch.float32)

    umi_output = umi_layer(example_data)
    print(f"\nInput shape: {example_data.shape}")
    print(f"Output shape: {umi_output.shape}")
    print(f"\nUMI Scores:")
    for i, score in enumerate(umi_output):
        print(f"  Sample {i + 1}: {score.item():.4f}")

    # Example 2: Alert Detection
    print("\n[Example 2] Alert Detection")
    print("-" * 60)

    critical_threshold = 1.0
    warning_threshold = 0.7

    alerts_critical = torch.abs(umi_output) > critical_threshold
    alerts_warning = torch.abs(umi_output) > warning_threshold

    print(f"Critical alerts (|UMI| > {critical_threshold}): {alerts_critical.tolist()}")
    print(f"Warning alerts (|UMI| > {warning_threshold}): {alerts_warning.tolist()}")

    # Example 3: Complete UMI Network
    print("\n[Example 3] Complete UMI Network with Preprocessing")
    print("-" * 60)

    umi_network = UMI_Network(hidden_dim=64, learnable_weights=False)
    print(f"UMI Network:\n{umi_network}")

    umi_scores, alert_levels = umi_network(example_data, return_alerts=True)

    print(f"\nProcessed UMI Scores:")
    alert_names = ['NORMAL', 'WARNING', 'CRITICAL']
    for i, (score, alert) in enumerate(zip(umi_scores, alert_levels)):
        print(f"  Sample {i + 1}: {score.item():7.4f} -> {alert_names[alert.item()]}")

    # Example 4: Learnable Weights
    print("\n[Example 4] Learnable UMI Weights")
    print("-" * 60)

    learnable_umi = UMI_Layer(learnable=True)
    print(f"Initial weights: {learnable_umi.get_weights()}")
    print(f"Trainable parameters: {sum(p.numel() for p in learnable_umi.parameters())}")

    # Example 5: Batch Processing
    print("\n[Example 5] Batch Processing")
    print("-" * 60)

    batch_size = 100
    random_data = torch.randn(batch_size, 4) * 0.5  # Random monitoring data

    batch_umi = umi_layer(random_data)
    print(f"Batch size: {batch_size}")
    print(f"Mean UMI: {batch_umi.mean().item():.4f}")
    print(f"Std UMI: {batch_umi.std().item():.4f}")
    print(f"Min UMI: {batch_umi.min().item():.4f}")
    print(f"Max UMI: {batch_umi.max().item():.4f}")

    print("\n" + "=" * 60)
    print("Migration from TensorFlow to PyTorch complete! ✅")
    print("=" * 60)
