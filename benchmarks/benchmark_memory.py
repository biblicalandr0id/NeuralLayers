"""
Memory Profiling for NeuralLayers

Analyzes:
- Peak memory usage
- Memory allocation patterns
- Memory efficiency across configurations
- Gradient memory requirements
"""

import torch
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
from pathlib import Path
import gc

import sys
sys.path.append('..')
from logicalbrain_network import UnifiedBrainNetwork


class MemoryProfiler:
    """Profile memory usage patterns"""

    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results = {}

        if device == 'cpu':
            print("⚠️  CPU mode - memory profiling will be limited")

    def profile_inference_memory(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        batch_sizes: List[int] = [1, 8, 32, 64, 128, 256]
    ) -> Dict:
        """Profile memory during inference"""

        print(f"\n{'='*70}")
        print(f"INFERENCE MEMORY PROFILING")
        print(f"{'='*70}")

        results = {
            'batch_sizes': batch_sizes,
            'peak_memory': [],
            'allocated_memory': [],
            'reserved_memory': []
        }

        model = UnifiedBrainNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=4
        ).to(self.device)
        model.eval()

        # Measure model size
        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        print(f"Model size: {model_size:.2f} MB")

        for batch_size in batch_sizes:
            # Clear cache
            if self.device == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                gc.collect()

            x = torch.randn(batch_size, input_dim).to(self.device)

            # Forward pass
            with torch.no_grad():
                output = model(x)

            if self.device == 'cuda':
                peak = torch.cuda.max_memory_allocated() / 1024**2
                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2

                results['peak_memory'].append(peak)
                results['allocated_memory'].append(allocated)
                results['reserved_memory'].append(reserved)

                print(f"Batch: {batch_size:4d} | Peak: {peak:8.2f} MB | "
                      f"Allocated: {allocated:8.2f} MB | Reserved: {reserved:8.2f} MB")

            # Clean up
            del x, output

        self.results['inference'] = results
        return results

    def profile_training_memory(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        batch_sizes: List[int] = [1, 8, 16, 32, 64]
    ) -> Dict:
        """Profile memory during training (with gradients)"""

        print(f"\n{'='*70}")
        print(f"TRAINING MEMORY PROFILING")
        print(f"{'='*70}")

        results = {
            'batch_sizes': batch_sizes,
            'forward_memory': [],
            'backward_memory': [],
            'peak_memory': [],
            'gradient_memory': []
        }

        for batch_size in batch_sizes:
            # Clear cache
            if self.device == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                gc.collect()

            model = UnifiedBrainNetwork(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=4
            ).to(self.device)
            model.train()

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            x = torch.randn(batch_size, input_dim).to(self.device)
            target = torch.randn(batch_size, input_dim).to(self.device)

            # Forward pass
            if self.device == 'cuda':
                torch.cuda.reset_peak_memory_stats()

            output = model(x)
            loss = torch.nn.functional.mse_loss(output['output'], target)

            if self.device == 'cuda':
                forward_mem = torch.cuda.max_memory_allocated() / 1024**2
                results['forward_memory'].append(forward_mem)

            # Backward pass
            if self.device == 'cuda':
                torch.cuda.reset_peak_memory_stats()

            loss.backward()

            if self.device == 'cuda':
                backward_mem = torch.cuda.max_memory_allocated() / 1024**2
                results['backward_memory'].append(backward_mem)

                # Total peak
                peak = torch.cuda.max_memory_allocated() / 1024**2
                results['peak_memory'].append(peak)

                # Gradient memory
                grad_mem = sum(
                    p.grad.numel() * p.grad.element_size()
                    for p in model.parameters() if p.grad is not None
                ) / 1024**2
                results['gradient_memory'].append(grad_mem)

                print(f"Batch: {batch_size:4d} | Forward: {forward_mem:8.2f} MB | "
                      f"Backward: {backward_mem:8.2f} MB | "
                      f"Peak: {peak:8.2f} MB | Gradients: {grad_mem:8.2f} MB")

            # Clean up
            del model, optimizer, x, target, output, loss

        self.results['training'] = results
        return results

    def profile_layer_memory(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        batch_size: int = 32
    ) -> Dict:
        """Profile memory per layer"""

        print(f"\n{'='*70}")
        print(f"LAYER-WISE MEMORY PROFILING")
        print(f"{'='*70}")

        if self.device == 'cpu':
            print("Skipping on CPU")
            return {}

        model = UnifiedBrainNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=4
        ).to(self.device)
        model.eval()

        x = torch.randn(batch_size, input_dim).to(self.device)

        layer_memory = {}

        # Hook to track activations
        activations = {}

        def get_activation_hook(name):
            def hook(module, input, output):
                if isinstance(output, dict):
                    # Calculate total size of all tensors in dict
                    total_size = 0
                    for v in output.values():
                        if isinstance(v, torch.Tensor):
                            total_size += v.numel() * v.element_size()
                    activations[name] = total_size / 1024**2
                elif isinstance(output, torch.Tensor):
                    activations[name] = output.numel() * output.element_size() / 1024**2
            return hook

        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(get_activation_hook(name))
                hooks.append(hook)

        # Forward pass
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            _ = model(x)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Print results
        print(f"\n{'Layer':<40} {'Memory (MB)':>12}")
        print("-" * 53)
        for name, mem in sorted(activations.items(), key=lambda x: x[1], reverse=True)[:20]:
            print(f"{name:<40} {mem:>12.2f}")

        self.results['layers'] = activations
        return activations

    def analyze_memory_efficiency(self) -> Dict:
        """Compute memory efficiency metrics"""

        print(f"\n{'='*70}")
        print(f"MEMORY EFFICIENCY ANALYSIS")
        print(f"{'='*70}")

        if 'inference' not in self.results or 'training' not in self.results:
            print("Run profiling first")
            return {}

        inf_data = self.results['inference']
        train_data = self.results['training']

        # Memory overhead (training vs inference)
        overhead = []
        batch_sizes = []

        for i, bs in enumerate(inf_data['batch_sizes']):
            if i < len(train_data['batch_sizes']):
                inf_mem = inf_data['peak_memory'][i]
                train_mem = train_data['peak_memory'][i]
                overhead.append((train_mem / inf_mem) - 1)
                batch_sizes.append(bs)

        avg_overhead = np.mean(overhead) * 100

        print(f"Average training memory overhead: {avg_overhead:.1f}%")

        # Memory per sample
        if inf_data['batch_sizes'][-1] > 1:
            largest_batch = inf_data['batch_sizes'][-1]
            idx = inf_data['batch_sizes'].index(largest_batch)
            mem_per_sample = inf_data['peak_memory'][idx] / largest_batch

            print(f"Memory per sample (inference): {mem_per_sample:.2f} MB")

        efficiency = {
            'training_overhead': avg_overhead,
            'memory_per_sample': mem_per_sample if 'mem_per_sample' in locals() else None
        }

        self.results['efficiency'] = efficiency
        return efficiency

    def plot_results(self, save_dir: str = 'benchmark_results'):
        """Generate visualization plots"""

        Path(save_dir).mkdir(exist_ok=True)

        if self.device == 'cpu':
            print("Skipping plots (CPU mode)")
            return

        # Plot 1: Inference vs Training Memory
        if 'inference' in self.results and 'training' in self.results:
            fig, ax = plt.subplots(figsize=(10, 6))

            inf = self.results['inference']
            train = self.results['training']

            ax.plot(inf['batch_sizes'], inf['peak_memory'],
                   'o-', label='Inference', linewidth=2)
            ax.plot(train['batch_sizes'], train['peak_memory'],
                   's-', label='Training', linewidth=2)

            ax.set_xlabel('Batch Size', fontsize=12)
            ax.set_ylabel('Peak Memory (MB)', fontsize=12)
            ax.set_title('Memory Usage: Inference vs Training', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'{save_dir}/memory_comparison.png', dpi=300, bbox_inches='tight')
            print(f"\nSaved memory comparison plot to {save_dir}/memory_comparison.png")

        # Plot 2: Training Memory Breakdown
        if 'training' in self.results:
            fig, ax = plt.subplots(figsize=(10, 6))

            train = self.results['training']

            ax.plot(train['batch_sizes'], train['forward_memory'],
                   'o-', label='Forward', linewidth=2)
            ax.plot(train['batch_sizes'], train['backward_memory'],
                   's-', label='Backward', linewidth=2)
            ax.plot(train['batch_sizes'], train['gradient_memory'],
                   '^-', label='Gradients', linewidth=2)

            ax.set_xlabel('Batch Size', fontsize=12)
            ax.set_ylabel('Memory (MB)', fontsize=12)
            ax.set_title('Training Memory Breakdown', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'{save_dir}/training_memory_breakdown.png', dpi=300, bbox_inches='tight')
            print(f"Saved training breakdown plot to {save_dir}/training_memory_breakdown.png")

        plt.close('all')

    def save_report(self, filepath: str = 'benchmark_results/MEMORY_REPORT.md'):
        """Generate markdown report"""

        Path(filepath).parent.mkdir(exist_ok=True)

        with open(filepath, 'w') as f:
            f.write("# NeuralLayers Memory Profiling Report\n\n")
            f.write(f"**Device**: {self.device.upper()}\n\n")

            # Inference section
            if 'inference' in self.results:
                f.write("## Inference Memory Usage\n\n")
                f.write("| Batch Size | Peak (MB) | Allocated (MB) | Reserved (MB) |\n")
                f.write("|------------|-----------|----------------|---------------|\n")

                data = self.results['inference']
                for i, bs in enumerate(data['batch_sizes']):
                    f.write(f"| {bs:4d} | {data['peak_memory'][i]:8.2f} | "
                           f"{data['allocated_memory'][i]:8.2f} | "
                           f"{data['reserved_memory'][i]:8.2f} |\n")
                f.write("\n")

            # Training section
            if 'training' in self.results:
                f.write("## Training Memory Usage\n\n")
                f.write("| Batch Size | Forward (MB) | Backward (MB) | Peak (MB) | Gradients (MB) |\n")
                f.write("|------------|--------------|---------------|-----------|----------------|\n")

                data = self.results['training']
                for i, bs in enumerate(data['batch_sizes']):
                    f.write(f"| {bs:4d} | {data['forward_memory'][i]:8.2f} | "
                           f"{data['backward_memory'][i]:8.2f} | "
                           f"{data['peak_memory'][i]:8.2f} | "
                           f"{data['gradient_memory'][i]:8.2f} |\n")
                f.write("\n")

            # Efficiency section
            if 'efficiency' in self.results:
                f.write("## Memory Efficiency\n\n")
                eff = self.results['efficiency']
                f.write(f"- **Training Overhead**: {eff['training_overhead']:.1f}%\n")
                if eff['memory_per_sample']:
                    f.write(f"- **Memory per Sample**: {eff['memory_per_sample']:.2f} MB\n")

        print(f"\nSaved memory report to {filepath}")


def main():
    """Run memory profiling"""

    print("🧠 NeuralLayers Memory Profiling Suite")
    print("=" * 70)

    profiler = MemoryProfiler()

    # Run profiling
    profiler.profile_inference_memory(batch_sizes=[1, 8, 16, 32, 64, 128])
    profiler.profile_training_memory(batch_sizes=[1, 8, 16, 32, 64])
    profiler.profile_layer_memory()
    profiler.analyze_memory_efficiency()

    # Generate visualizations and report
    profiler.plot_results()
    profiler.save_report()

    print("\n✅ Memory profiling completed!")


if __name__ == '__main__':
    main()
