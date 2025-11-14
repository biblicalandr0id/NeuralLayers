"""
Hyperparameter Search and AutoML for NeuralLayers

Features:
- Optuna integration for Bayesian optimization
- Neural Architecture Search (NAS)
- Learning rate finder
- Automated hyperparameter tuning
- Multi-objective optimization
- Pruning of unpromising trials
- Visualization of optimization results
"""

import os
import json
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


@dataclass
class SearchConfig:
    """Configuration for hyperparameter search"""

    # Optimization
    n_trials: int = 100
    timeout: Optional[int] = None  # Seconds
    study_name: str = "neurallayers_optimization"
    storage: Optional[str] = None  # SQLite URL for persistence

    # Pruning
    use_pruning: bool = True
    pruning_warmup_steps: int = 5

    # Multi-objective
    directions: List[str] = None  # e.g., ["minimize", "maximize"]

    # Parallelization
    n_jobs: int = 1  # Number of parallel trials

    # Search space
    search_space: Dict[str, Any] = None

    # Output
    save_best_model: bool = True
    output_dir: str = "hyperparameter_search"


class LearningRateFinder:
    """
    Learning Rate Range Test (LR Finder)

    Based on: "Cyclical Learning Rates for Training Neural Networks"
    (Smith, 2017)

    Finds optimal learning rate by gradually increasing LR and
    monitoring loss.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = "cpu"
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        # Save initial state
        self.initial_state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

    def range_test(
        self,
        train_loader: torch.utils.data.DataLoader,
        start_lr: float = 1e-7,
        end_lr: float = 10,
        num_iter: int = 100,
        smooth_f: float = 0.05
    ) -> Dict[str, List[float]]:
        """
        Run learning rate range test

        Args:
            train_loader: Training data loader
            start_lr: Starting learning rate
            end_lr: Ending learning rate
            num_iter: Number of iterations
            smooth_f: Smoothing factor for loss

        Returns:
            Dictionary with 'lrs' and 'losses' lists
        """

        # LR schedule (exponential)
        mult = (end_lr / start_lr) ** (1 / num_iter)

        lrs = []
        losses = []
        best_loss = float('inf')

        # Set initial LR
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = start_lr

        lr = start_lr
        avg_loss = 0.0
        beta = 1 - smooth_f

        self.model.train()

        iterator = iter(train_loader)

        for iteration in range(num_iter):
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                inputs, targets = next(iterator)

            # Move to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            # Handle dict output
            if isinstance(outputs, dict):
                outputs = outputs['output']

            loss = self.criterion(outputs, targets)

            # Check for explosion
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Loss exploded at LR={lr:.2e}")
                break

            # Smooth loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta ** (iteration + 1))

            # Track best
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss

            # Stop if loss is exploding (4x best loss)
            if smoothed_loss > 4 * best_loss:
                print(f"Stopping early at LR={lr:.2e} (loss explosion)")
                break

            # Record
            lrs.append(lr)
            losses.append(smoothed_loss)

            # Backward
            loss.backward()
            self.optimizer.step()

            # Update LR
            lr *= mult
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        # Restore initial state
        self.model.load_state_dict(self.initial_state['model'])
        self.optimizer.load_state_dict(self.initial_state['optimizer'])

        return {'lrs': lrs, 'losses': losses}

    def suggest_lr(self, lrs: List[float], losses: List[float]) -> float:
        """
        Suggest optimal learning rate

        Uses the point with steepest negative gradient
        """

        # Compute gradients
        gradients = np.gradient(losses)

        # Find steepest point
        min_grad_idx = np.argmin(gradients)

        # Suggested LR is ~10x before the steepest point
        suggested_idx = max(0, min_grad_idx - int(0.1 * len(lrs)))

        return lrs[suggested_idx]

    def plot(self, lrs: List[float], losses: List[float], suggested_lr: float = None):
        """Plot learning rate range test results"""

        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            plt.plot(lrs, losses)
            plt.xscale('log')
            plt.xlabel('Learning Rate')
            plt.ylabel('Loss')
            plt.title('Learning Rate Range Test')

            if suggested_lr:
                plt.axvline(suggested_lr, color='red', linestyle='--', label=f'Suggested LR: {suggested_lr:.2e}')
                plt.legend()

            plt.grid(True)
            plt.show()

        except ImportError:
            print("⚠️  matplotlib not installed, cannot plot")


class OptunaOptimizer:
    """
    Hyperparameter optimization using Optuna

    Supports:
    - Bayesian optimization (TPE sampler)
    - Multi-objective optimization
    - Pruning of unpromising trials
    - Distributed optimization
    """

    def __init__(self, config: SearchConfig):
        self.config = config

    def create_study(self, directions: Optional[List[str]] = None):
        """Create Optuna study"""

        try:
            import optuna

            # Pruner
            if self.config.use_pruning:
                pruner = optuna.pruners.MedianPruner(
                    n_warmup_steps=self.config.pruning_warmup_steps
                )
            else:
                pruner = optuna.pruners.NopPruner()

            # Sampler
            sampler = optuna.samplers.TPESampler()

            # Create study
            if directions:
                # Multi-objective
                study = optuna.create_study(
                    study_name=self.config.study_name,
                    storage=self.config.storage,
                    sampler=sampler,
                    pruner=pruner,
                    directions=directions,
                    load_if_exists=True
                )
            else:
                # Single-objective
                study = optuna.create_study(
                    study_name=self.config.study_name,
                    storage=self.config.storage,
                    sampler=sampler,
                    pruner=pruner,
                    direction="minimize",
                    load_if_exists=True
                )

            return study

        except ImportError:
            raise RuntimeError("optuna not installed. Install with: pip install optuna")

    def optimize(
        self,
        objective: Callable,
        directions: Optional[List[str]] = None
    ):
        """
        Run hyperparameter optimization

        Args:
            objective: Objective function that takes an Optuna trial and returns metric(s)
            directions: List of optimization directions for multi-objective

        Returns:
            Optuna study object
        """

        import optuna

        # Create study
        study = self.create_study(directions)

        print("="*80)
        print("HYPERPARAMETER OPTIMIZATION")
        print("="*80)
        print(f"Study name: {self.config.study_name}")
        print(f"Number of trials: {self.config.n_trials}")
        print(f"Timeout: {self.config.timeout}s" if self.config.timeout else "Timeout: None")
        print("="*80)

        # Optimize
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs,
            show_progress_bar=True
        )

        # Print results
        print("\n" + "="*80)
        print("OPTIMIZATION RESULTS")
        print("="*80)

        if directions:
            # Multi-objective
            print(f"Number of finished trials: {len(study.trials)}")
            print(f"Pareto front size: {len(study.best_trials)}")

            for i, trial in enumerate(study.best_trials):
                print(f"\nPareto Trial {i+1}:")
                print(f"  Values: {trial.values}")
                print(f"  Params: {trial.params}")

        else:
            # Single-objective
            print(f"Number of finished trials: {len(study.trials)}")
            print(f"Best trial:")
            print(f"  Value: {study.best_value}")
            print(f"  Params: {study.best_params}")

        print("="*80)

        return study

    def visualize_optimization(self, study):
        """Visualize optimization results"""

        try:
            import optuna
            import matplotlib.pyplot as plt

            # 1. Optimization history
            fig1 = optuna.visualization.matplotlib.plot_optimization_history(study)
            plt.show()

            # 2. Parameter importances
            fig2 = optuna.visualization.matplotlib.plot_param_importances(study)
            plt.show()

            # 3. Parallel coordinate plot
            fig3 = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
            plt.show()

        except ImportError:
            print("⚠️  optuna or matplotlib not installed for visualization")


class NeuralArchitectureSearch:
    """
    Neural Architecture Search (NAS)

    Searches for optimal model architecture
    """

    def __init__(self, config: SearchConfig):
        self.config = config

    def search_space_objective(
        self,
        trial,
        train_fn: Callable,
        model_builder: Callable
    ) -> float:
        """
        Objective function for architecture search

        Args:
            trial: Optuna trial
            train_fn: Function that trains model and returns validation metric
            model_builder: Function that builds model from hyperparameters

        Returns:
            Validation metric
        """

        # Suggest hyperparameters
        hyperparams = {}

        # Model architecture
        hyperparams['hidden_dim'] = trial.suggest_int('hidden_dim', 64, 512, step=64)
        hyperparams['num_layers'] = trial.suggest_int('num_layers', 2, 8)
        hyperparams['num_heads'] = trial.suggest_categorical('num_heads', [4, 8, 16])
        hyperparams['dropout'] = trial.suggest_float('dropout', 0.0, 0.5)

        # Training hyperparameters
        hyperparams['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        hyperparams['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        hyperparams['optimizer'] = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd'])
        hyperparams['scheduler'] = trial.suggest_categorical('scheduler', ['cosine', 'linear', 'constant'])

        # Build model
        model = model_builder(hyperparams)

        # Train and evaluate
        val_metric = train_fn(model, hyperparams, trial)

        return val_metric


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from logicalbrain_network import UnifiedBrainLogicNetwork
    from data_pipeline import create_synthetic_dataset, NeuralLayersDataset
    from torch.utils.data import DataLoader

    print("="*80)
    print("HYPERPARAMETER SEARCH DEMO")
    print("="*80)

    # 1. Learning Rate Finder
    print("\n1. Learning Rate Finder")
    print("-"*80)

    # Create model and data
    model = UnifiedBrainLogicNetwork(128, 128, 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    X_train, y_train = create_synthetic_dataset(num_samples=100, input_dim=128, output_dim=64)
    train_dataset = NeuralLayersDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Run LR finder
    lr_finder = LearningRateFinder(model, optimizer, criterion)
    results = lr_finder.range_test(
        train_loader,
        start_lr=1e-7,
        end_lr=1,
        num_iter=50
    )

    suggested_lr = lr_finder.suggest_lr(results['lrs'], results['losses'])
    print(f"Suggested learning rate: {suggested_lr:.2e}")

    # 2. Optuna Optimization
    print("\n2. Optuna Optimization")
    print("-"*80)

    try:
        import optuna

        def objective(trial):
            """Simple objective function"""

            # Suggest hyperparameters
            lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            hidden_dim = trial.suggest_int('hidden_dim', 64, 256, step=64)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

            # Create model
            model = UnifiedBrainLogicNetwork(128, hidden_dim, 64)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # Train for a few steps
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            model.train()
            total_loss = 0.0

            for i, (inputs, targets) in enumerate(train_loader):
                if i >= 5:  # Quick training for demo
                    break

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = nn.functional.mse_loss(outputs['output'], targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / 5

            # Report intermediate value for pruning
            trial.report(avg_loss, step=0)

            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()

            return avg_loss

        # Create config
        config = SearchConfig(
            n_trials=10,
            study_name="demo_optimization",
            use_pruning=True
        )

        # Run optimization
        optimizer = OptunaOptimizer(config)
        study = optimizer.optimize(objective)

        print(f"\n✅ Best hyperparameters: {study.best_params}")
        print(f"✅ Best value: {study.best_value:.4f}")

    except ImportError:
        print("⚠️  Optuna not installed. Install with: pip install optuna")

    print("\n" + "="*80)
    print("✅ Hyperparameter search demo complete!")
    print("="*80)
