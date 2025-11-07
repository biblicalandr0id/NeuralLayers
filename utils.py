"""
NeuralLayers Utilities

Comprehensive utility functions for:
- Configuration management
- Logging and debugging
- Model checkpointing and serialization
- Visualization
- Benchmarking and profiling
- Numerical stability
- GPU acceleration

Author: biblicalandr0id
"""

import os
import yaml
import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import json
import time


# =============================================================================
# Configuration Management
# =============================================================================

class Config:
    """Configuration manager for NeuralLayers."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to YAML config file. If None, uses default config.
        """
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'model': {
                'input_dim': 1024,
                'hidden_dim': 2048,
                'output_dim': 512
            },
            'brain': {
                'V_rest': -70.0,
                'V_peak': 40.0,
                'tau_membrane': 20.0
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 0.001,
                'num_epochs': 100
            },
            'device': {
                'use_cuda': True,
                'cuda_device': 0
            },
            'logging': {
                'level': 'INFO',
                'log_to_file': True
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'model.input_dim')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def save(self, path: str):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)


# =============================================================================
# Logging Infrastructure
# =============================================================================

class Logger:
    """Enhanced logger for NeuralLayers."""

    def __init__(self, name: str = "NeuralLayers", config: Optional[Config] = None):
        """
        Initialize logger.

        Args:
            name: Logger name
            config: Configuration object
        """
        self.config = config or Config()
        self.logger = logging.getLogger(name)

        # Set level
        level_str = self.config.get('logging.level', 'INFO')
        level = getattr(logging, level_str)
        self.logger.setLevel(level)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        formatter = logging.Formatter(
            '[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler
        if self.config.get('logging.log_to_file', False):
            log_file = self.config.get('logging.log_file', 'neurallayers.log')
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)

    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)


# =============================================================================
# Model Checkpointing and Serialization
# =============================================================================

class CheckpointManager:
    """Manage model checkpoints."""

    def __init__(self, save_dir: str = "./checkpoints", keep_last_n: int = 5):
        """
        Initialize checkpoint manager.

        Args:
            save_dir: Directory to save checkpoints
            keep_last_n: Keep only last N checkpoints
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.checkpoints = []

    def save(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None,
             epoch: int = 0, step: int = 0, metrics: Optional[Dict] = None,
             **kwargs):
        """
        Save model checkpoint.

        Args:
            model: PyTorch model
            optimizer: Optimizer (optional)
            epoch: Current epoch
            step: Current step
            metrics: Dictionary of metrics
            **kwargs: Additional data to save
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_epoch{epoch}_step{step}_{timestamp}.pt"
        filepath = self.save_dir / filename

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'step': step,
            'timestamp': timestamp,
        }

        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        if metrics:
            checkpoint['metrics'] = metrics

        # Add any additional kwargs
        checkpoint.update(kwargs)

        torch.save(checkpoint, filepath)
        self.checkpoints.append(filepath)

        # Remove old checkpoints
        if len(self.checkpoints) > self.keep_last_n:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()

        return filepath

    def load(self, filepath: str, model: nn.Module,
             optimizer: Optional[torch.optim.Optimizer] = None):
        """
        Load model checkpoint.

        Args:
            filepath: Path to checkpoint file
            model: PyTorch model
            optimizer: Optimizer (optional)

        Returns:
            Dictionary containing checkpoint data
        """
        checkpoint = torch.load(filepath)

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint

    def get_latest(self) -> Optional[Path]:
        """Get latest checkpoint path."""
        if not self.checkpoints:
            # Check directory for existing checkpoints
            checkpoints = sorted(self.save_dir.glob("checkpoint_*.pt"))
            if checkpoints:
                return checkpoints[-1]
            return None
        return self.checkpoints[-1]


# =============================================================================
# Visualization Tools
# =============================================================================

class StateVisualizer:
    """Visualize network states."""

    def __init__(self, save_dir: str = "./plots"):
        """
        Initialize visualizer.

        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_system_state(self, state: Dict[str, torch.Tensor], step: int = 0):
        """
        Plot system state variables.

        Args:
            state: Dictionary of state variables
            step: Current step number
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            components = ['V', 'NT', 'Ca', 'ATP', 'g', 'Ψ', 'τ', 'ω']
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            axes = axes.flatten()

            for idx, component in enumerate(components):
                if component in state:
                    values = state[component].detach().cpu().numpy()
                    if values.ndim > 1:
                        values = values.mean(axis=0)

                    axes[idx].plot(values)
                    axes[idx].set_title(f'{component} (Step {step})')
                    axes[idx].grid(True)

            plt.tight_layout()
            plt.savefig(self.save_dir / f'system_state_step{step}.png', dpi=150)
            plt.close()

        except ImportError:
            print("Warning: matplotlib not installed. Skipping visualization.")

    def plot_training_metrics(self, metrics: Dict[str, List[float]]):
        """
        Plot training metrics.

        Args:
            metrics: Dictionary of metric lists
        """
        try:
            import matplotlib.pyplot as plt

            num_metrics = len(metrics)
            fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 4))

            if num_metrics == 1:
                axes = [axes]

            for idx, (name, values) in enumerate(metrics.items()):
                axes[idx].plot(values)
                axes[idx].set_title(name)
                axes[idx].set_xlabel('Step')
                axes[idx].grid(True)

            plt.tight_layout()
            plt.savefig(self.save_dir / 'training_metrics.png', dpi=150)
            plt.close()

        except ImportError:
            print("Warning: matplotlib not installed. Skipping visualization.")


# =============================================================================
# Benchmarking and Profiling
# =============================================================================

class Profiler:
    """Profiling utilities for performance analysis."""

    def __init__(self):
        """Initialize profiler."""
        self.timings = {}
        self.start_times = {}

    def start(self, name: str):
        """Start timing a section."""
        self.start_times[name] = time.time()

    def end(self, name: str):
        """End timing a section."""
        if name in self.start_times:
            elapsed = time.time() - self.start_times[name]
            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(elapsed)
            del self.start_times[name]
            return elapsed
        return None

    def report(self) -> str:
        """Generate profiling report."""
        lines = ["=== Profiling Report ==="]
        for name, times in self.timings.items():
            mean_time = sum(times) / len(times)
            total_time = sum(times)
            lines.append(f"{name}:")
            lines.append(f"  Mean: {mean_time * 1000:.2f} ms")
            lines.append(f"  Total: {total_time:.2f} s")
            lines.append(f"  Calls: {len(times)}")
        return "\n".join(lines)

    def reset(self):
        """Reset profiler."""
        self.timings.clear()
        self.start_times.clear()


# =============================================================================
# Numerical Stability
# =============================================================================

class GradientClipper:
    """Gradient clipping for numerical stability."""

    def __init__(self, clip_value: float = 1.0, clip_norm: Optional[float] = None):
        """
        Initialize gradient clipper.

        Args:
            clip_value: Clip gradients by value
            clip_norm: Clip gradients by norm (optional)
        """
        self.clip_value = clip_value
        self.clip_norm = clip_norm

    def clip(self, model: nn.Module):
        """
        Clip model gradients.

        Args:
            model: PyTorch model
        """
        if self.clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_norm)
        elif self.clip_value:
            torch.nn.utils.clip_grad_value_(model.parameters(), self.clip_value)


# =============================================================================
# GPU Acceleration
# =============================================================================

class DeviceManager:
    """Manage device placement (CPU/GPU)."""

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize device manager.

        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self._device = self._setup_device()

    def _setup_device(self) -> torch.device:
        """Setup computing device."""
        use_cuda = self.config.get('device.use_cuda', True)
        if use_cuda and torch.cuda.is_available():
            device_id = self.config.get('device.cuda_device', 0)
            device = torch.device(f'cuda:{device_id}')
            print(f"✅ Using GPU: {torch.cuda.get_device_name(device_id)}")
        else:
            device = torch.device('cpu')
            print("ℹ️  Using CPU")
        return device

    @property
    def device(self) -> torch.device:
        """Get current device."""
        return self._device

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to device."""
        return tensor.to(self._device)

    def model_to_device(self, model: nn.Module) -> nn.Module:
        """Move model to device."""
        return model.to(self._device)


# =============================================================================
# Input Validation
# =============================================================================

class InputValidator:
    """Validate input tensors."""

    @staticmethod
    def validate_shape(tensor: torch.Tensor, expected_shape: tuple,
                       name: str = "tensor"):
        """
        Validate tensor shape.

        Args:
            tensor: Input tensor
            expected_shape: Expected shape (use -1 for any dimension)
            name: Tensor name for error messages

        Raises:
            ValueError: If shape doesn't match
        """
        actual_shape = tensor.shape
        if len(actual_shape) != len(expected_shape):
            raise ValueError(
                f"{name} has {len(actual_shape)} dimensions, "
                f"expected {len(expected_shape)}"
            )

        for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
            if expected != -1 and actual != expected:
                raise ValueError(
                    f"{name} dimension {i} is {actual}, expected {expected}"
                )

    @staticmethod
    def validate_range(tensor: torch.Tensor, min_val: float, max_val: float,
                       name: str = "tensor"):
        """
        Validate tensor value range.

        Args:
            tensor: Input tensor
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            name: Tensor name for error messages

        Raises:
            ValueError: If values out of range
        """
        if tensor.min() < min_val or tensor.max() > max_val:
            raise ValueError(
                f"{name} has values outside [{min_val}, {max_val}] range: "
                f"[{tensor.min():.4f}, {tensor.max():.4f}]"
            )

    @staticmethod
    def check_nan_inf(tensor: torch.Tensor, name: str = "tensor"):
        """
        Check for NaN or Inf values.

        Args:
            tensor: Input tensor
            name: Tensor name for error messages

        Raises:
            ValueError: If NaN or Inf found
        """
        if torch.isnan(tensor).any():
            raise ValueError(f"{name} contains NaN values")
        if torch.isinf(tensor).any():
            raise ValueError(f"{name} contains Inf values")


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NeuralLayers Utilities - Examples")
    print("=" * 60)

    # Configuration
    print("\n[1] Configuration Management")
    config = Config('config.yaml')
    print(f"Model input dim: {config.get('model.input_dim')}")
    print(f"Batch size: {config.get('training.batch_size')}")

    # Logging
    print("\n[2] Logging")
    logger = Logger(config=config)
    logger.info("This is an info message")
    logger.warning("This is a warning")

    # Device Management
    print("\n[3] Device Management")
    device_manager = DeviceManager(config)
    print(f"Device: {device_manager.device}")

    # Profiling
    print("\n[4] Profiling")
    profiler = Profiler()
    profiler.start("test_operation")
    time.sleep(0.1)
    profiler.end("test_operation")
    print(profiler.report())

    # Input Validation
    print("\n[5] Input Validation")
    validator = InputValidator()
    tensor = torch.randn(32, 1024)
    validator.validate_shape(tensor, (32, 1024), "input_tensor")
    validator.check_nan_inf(tensor, "input_tensor")
    print("✅ Validation passed")

    print("\n" + "=" * 60)
