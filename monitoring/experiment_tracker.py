"""
Comprehensive Experiment Tracking for NeuralLayers

Features:
- Weights & Biases integration
- TensorBoard logging
- MLflow tracking
- Metric aggregation
- Real-time visualization
- Experiment comparison
- Model artifact tracking
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
import json
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking"""

    # Experiment metadata
    experiment_name: str = "neurallayers_experiment"
    run_name: Optional[str] = None
    tags: List[str] = None
    notes: Optional[str] = None

    # Tracking backends
    use_wandb: bool = True
    use_tensorboard: bool = True
    use_mlflow: bool = False

    # Logging
    log_interval: int = 10  # Log every N steps
    log_gradients: bool = True
    log_model_architecture: bool = True
    log_system_metrics: bool = True

    # Artifacts
    save_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints"
    save_predictions: bool = False
    max_predictions_to_save: int = 100

    # Visualization
    plot_interval: int = 100
    max_plots: int = 10


class BaseTracker:
    """Base class for experiment trackers"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.run_id = None

    def init(self, run_config: Dict[str, Any]):
        """Initialize tracker"""
        raise NotImplementedError

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics"""
        raise NotImplementedError

    def log_artifact(self, artifact_path: str, artifact_type: str = "file"):
        """Log artifact"""
        raise NotImplementedError

    def finish(self):
        """Finish tracking"""
        raise NotImplementedError


class WandBTracker(BaseTracker):
    """Weights & Biases tracker"""

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.run = None

    def init(self, run_config: Dict[str, Any]):
        """Initialize W&B"""

        try:
            import wandb

            self.run = wandb.init(
                project=self.config.experiment_name,
                name=self.config.run_name,
                config=run_config,
                tags=self.config.tags,
                notes=self.config.notes
            )

            self.run_id = self.run.id

            print(f"✅ W&B initialized: {self.run.url}")

        except ImportError:
            print("⚠️  wandb not installed, skipping W&B tracking")

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to W&B"""

        if self.run:
            import wandb
            wandb.log(metrics, step=step)

    def log_model(self, model: nn.Module, name: str = "model"):
        """Log model architecture"""

        if self.run:
            import wandb

            # Log model architecture as text
            wandb.config.update({
                "model_architecture": str(model),
                "model_parameters": sum(p.numel() for p in model.parameters())
            })

    def log_artifact(self, artifact_path: str, artifact_type: str = "file"):
        """Log artifact to W&B"""

        if self.run:
            import wandb

            artifact = wandb.Artifact(
                name=f"{self.config.experiment_name}_{artifact_type}",
                type=artifact_type
            )
            artifact.add_file(artifact_path)
            self.run.log_artifact(artifact)

    def log_plot(self, figure, name: str, step: int):
        """Log matplotlib figure"""

        if self.run:
            import wandb
            wandb.log({name: wandb.Image(figure)}, step=step)

    def finish(self):
        """Finish W&B run"""

        if self.run:
            import wandb
            wandb.finish()


class TensorBoardTracker(BaseTracker):
    """TensorBoard tracker"""

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.writer = None

    def init(self, run_config: Dict[str, Any]):
        """Initialize TensorBoard"""

        try:
            from torch.utils.tensorboard import SummaryWriter

            log_dir = Path("runs") / self.config.experiment_name

            if self.config.run_name:
                log_dir = log_dir / self.config.run_name
            else:
                log_dir = log_dir / datetime.now().strftime("%Y%m%d-%H%M%S")

            self.writer = SummaryWriter(log_dir=str(log_dir))

            # Log config as text
            config_text = json.dumps(run_config, indent=2)
            self.writer.add_text("config", config_text, 0)

            print(f"✅ TensorBoard initialized: {log_dir}")

        except ImportError:
            print("⚠️  tensorboard not installed, skipping TensorBoard tracking")

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to TensorBoard"""

        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)

    def log_model_graph(self, model: nn.Module, input_shape: tuple):
        """Log model graph"""

        if self.writer:
            dummy_input = torch.randn(*input_shape)
            self.writer.add_graph(model, dummy_input)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        """Log histogram"""

        if self.writer:
            self.writer.add_histogram(tag, values, step)

    def log_image(self, tag: str, image: np.ndarray, step: int):
        """Log image"""

        if self.writer:
            self.writer.add_image(tag, image, step, dataformats='HWC')

    def finish(self):
        """Finish TensorBoard logging"""

        if self.writer:
            self.writer.close()


class MLflowTracker(BaseTracker):
    """MLflow tracker"""

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.run = None

    def init(self, run_config: Dict[str, Any]):
        """Initialize MLflow"""

        try:
            import mlflow

            mlflow.set_experiment(self.config.experiment_name)

            self.run = mlflow.start_run(run_name=self.config.run_name)

            # Log parameters
            for key, value in run_config.items():
                mlflow.log_param(key, value)

            # Log tags
            if self.config.tags:
                for tag in self.config.tags:
                    mlflow.set_tag("tag", tag)

            self.run_id = self.run.info.run_id

            print(f"✅ MLflow initialized: {self.run_id}")

        except ImportError:
            print("⚠️  mlflow not installed, skipping MLflow tracking")

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to MLflow"""

        if self.run:
            import mlflow

            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)

    def log_artifact(self, artifact_path: str, artifact_type: str = "file"):
        """Log artifact to MLflow"""

        if self.run:
            import mlflow
            mlflow.log_artifact(artifact_path)

    def log_model(self, model: nn.Module, name: str = "model"):
        """Log model to MLflow"""

        if self.run:
            import mlflow.pytorch
            mlflow.pytorch.log_model(model, name)

    def finish(self):
        """Finish MLflow run"""

        if self.run:
            import mlflow
            mlflow.end_run()


class ExperimentTracker:
    """
    Unified experiment tracker supporting multiple backends

    Supports:
    - Weights & Biases
    - TensorBoard
    - MLflow
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.trackers: List[BaseTracker] = []

        # Initialize trackers based on config
        if config.use_wandb:
            self.trackers.append(WandBTracker(config))

        if config.use_tensorboard:
            self.trackers.append(TensorBoardTracker(config))

        if config.use_mlflow:
            self.trackers.append(MLflowTracker(config))

        self.step_count = 0
        self.start_time = time.time()

    def init(self, run_config: Dict[str, Any]):
        """Initialize all trackers"""

        for tracker in self.trackers:
            tracker.init(run_config)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = ""
    ):
        """Log metrics to all trackers"""

        if step is None:
            step = self.step_count
            self.step_count += 1

        # Add prefix if provided
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        # Log to all trackers
        for tracker in self.trackers:
            tracker.log_metrics(metrics, step)

    def log_model(self, model: nn.Module, name: str = "model"):
        """Log model to all trackers"""

        for tracker in self.trackers:
            if hasattr(tracker, 'log_model'):
                tracker.log_model(model, name)

    def log_gradients(self, model: nn.Module, step: int):
        """Log gradient statistics"""

        if not self.config.log_gradients:
            return

        grad_metrics = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach()

                grad_metrics[f"gradients/{name}/mean"] = grad.mean().item()
                grad_metrics[f"gradients/{name}/std"] = grad.std().item()
                grad_metrics[f"gradients/{name}/max"] = grad.max().item()
                grad_metrics[f"gradients/{name}/min"] = grad.min().item()
                grad_metrics[f"gradients/{name}/norm"] = grad.norm().item()

        self.log_metrics(grad_metrics, step)

    def log_parameters(self, model: nn.Module, step: int):
        """Log parameter statistics"""

        param_metrics = {}

        for name, param in model.named_parameters():
            data = param.detach()

            param_metrics[f"parameters/{name}/mean"] = data.mean().item()
            param_metrics[f"parameters/{name}/std"] = data.std().item()
            param_metrics[f"parameters/{name}/max"] = data.max().item()
            param_metrics[f"parameters/{name}/min"] = data.min().item()
            param_metrics[f"parameters/{name}/norm"] = data.norm().item()

        self.log_metrics(param_metrics, step)

    def log_system_metrics(self, step: int):
        """Log system metrics (CPU, GPU, memory)"""

        if not self.config.log_system_metrics:
            return

        import psutil

        system_metrics = {
            "system/cpu_percent": psutil.cpu_percent(),
            "system/memory_percent": psutil.virtual_memory().percent,
            "system/disk_percent": psutil.disk_usage('/').percent
        }

        # GPU metrics if available
        if torch.cuda.is_available():
            system_metrics.update({
                "system/gpu_memory_allocated": torch.cuda.memory_allocated() / 1e9,  # GB
                "system/gpu_memory_reserved": torch.cuda.memory_reserved() / 1e9,  # GB
                "system/gpu_utilization": torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
            })

        self.log_metrics(system_metrics, step)

    def log_learning_rate(self, optimizer: torch.optim.Optimizer, step: int):
        """Log current learning rate"""

        lrs = {}
        for i, param_group in enumerate(optimizer.param_groups):
            lrs[f"learning_rate/group_{i}"] = param_group['lr']

        self.log_metrics(lrs, step)

    def log_artifact(self, artifact_path: str, artifact_type: str = "file"):
        """Log artifact to all trackers"""

        for tracker in self.trackers:
            tracker.log_artifact(artifact_path, artifact_type)

    def log_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: List[str],
        step: int
    ):
        """Log confusion matrix"""

        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import confusion_matrix
            import seaborn as sns

            cm = confusion_matrix(y_true, y_pred)

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=labels,
                yticklabels=labels,
                ax=ax
            )
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title('Confusion Matrix')

            # Log to W&B if available
            for tracker in self.trackers:
                if isinstance(tracker, WandBTracker):
                    tracker.log_plot(fig, "confusion_matrix", step)

            plt.close(fig)

        except ImportError:
            print("⚠️  matplotlib/seaborn/sklearn not installed, skipping confusion matrix")

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ):
        """Save model checkpoint"""

        if not self.config.save_checkpoints:
            return

        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }

        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)

            # Log as artifact
            self.log_artifact(str(best_path), artifact_type="model")

        print(f"💾 Checkpoint saved: {checkpoint_path}")

    def get_runtime(self) -> float:
        """Get total runtime in seconds"""
        return time.time() - self.start_time

    def finish(self):
        """Finish all trackers"""

        runtime = self.get_runtime()

        print(f"\n{'='*80}")
        print(f"Experiment Complete")
        print(f"{'='*80}")
        print(f"Total runtime: {runtime:.2f}s ({runtime/60:.2f}m)")
        print(f"Total steps: {self.step_count}")
        print(f"{'='*80}")

        for tracker in self.trackers:
            tracker.finish()


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from logicalbrain_network import UnifiedBrainLogicNetwork

    # Create experiment config
    config = ExperimentConfig(
        experiment_name="test_experiment",
        run_name="test_run_1",
        tags=["test", "demo"],
        use_wandb=False,  # Set to True if wandb is installed
        use_tensorboard=True,
        log_gradients=True,
        log_system_metrics=True
    )

    # Initialize tracker
    tracker = ExperimentTracker(config)

    # Run config
    run_config = {
        "learning_rate": 1e-4,
        "batch_size": 32,
        "optimizer": "adamw",
        "model": "UnifiedBrainLogicNetwork"
    }

    tracker.init(run_config)

    # Create model
    model = UnifiedBrainLogicNetwork(
        input_dim=128,
        hidden_dim=128,
        output_dim=64
    )

    # Log model
    tracker.log_model(model)

    # Simulate training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print("="*80)
    print("EXPERIMENT TRACKING DEMO")
    print("="*80)

    for epoch in range(5):
        for step in range(10):
            # Dummy training step
            x = torch.randn(4, 128)
            output = model(x)
            loss = output['output'].sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log metrics
            metrics = {
                "loss": loss.item(),
                "epoch": epoch
            }

            global_step = epoch * 10 + step

            tracker.log_metrics(metrics, step=global_step, prefix="train")
            tracker.log_gradients(model, global_step)
            tracker.log_learning_rate(optimizer, global_step)

            if global_step % 5 == 0:
                tracker.log_system_metrics(global_step)

        print(f"Epoch {epoch+1}/5 complete")

        # Save checkpoint
        tracker.save_checkpoint(
            model,
            optimizer,
            epoch,
            metrics,
            is_best=(epoch == 4)
        )

    # Finish tracking
    tracker.finish()

    print("\n✅ Experiment tracking demo complete!")
