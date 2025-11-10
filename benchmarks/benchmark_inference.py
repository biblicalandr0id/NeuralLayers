"""
Inference Benchmarking for NeuralLayers

Measures:
- Throughput (samples/sec)
- Latency (ms/sample)
- GPU memory usage
- Batch size scaling
"""

import time
import torch
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.append('..')
from logicalbrain_network import UnifiedBrainNetwork


class InferenceBenchmark:
    """Benchmark inference performance"""

    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results = {}

    def benchmark_throughput(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        batch_sizes: List[int] = [1, 4, 16, 32, 64, 128],
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict[str, List[float]]:
        """Measure throughput across different batch sizes"""

        print(f"\n{'='*70}")
        print(f"THROUGHPUT BENCHMARK - {self.device.upper()}")
        print(f"{'='*70}")

        model = UnifiedBrainNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=4
        ).to(self.device)
        model.eval()

        throughputs = []
        latencies = []
        memory_usage = []

        for batch_size in batch_sizes:
            # Create dummy input
            x = torch.randn(batch_size, input_dim).to(self.device)

            # Warmup
            with torch.no_grad():
                for _ in range(warmup_iterations):
                    _ = model(x)

            # Synchronize GPU
            if self.device == 'cuda':
                torch.cuda.synchronize()

            # Benchmark
            times = []
            with torch.no_grad():
                for _ in range(num_iterations):
                    start = time.perf_counter()
                    _ = model(x)

                    if self.device == 'cuda':
                        torch.cuda.synchronize()

                    end = time.perf_counter()
                    times.append(end - start)

            # Calculate metrics
            avg_time = np.mean(times)
            throughput = batch_size / avg_time
            latency = (avg_time / batch_size) * 1000  # ms per sample

            throughputs.append(throughput)
            latencies.append(latency)

            # Memory usage
            if self.device == 'cuda':
                mem_allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
                memory_usage.append(mem_allocated)
                torch.cuda.reset_peak_memory_stats()
            else:
                memory_usage.append(0)

            print(f"Batch size: {batch_size:4d} | "
                  f"Throughput: {throughput:8.2f} samples/sec | "
                  f"Latency: {latency:6.2f} ms/sample | "
                  f"Memory: {memory_usage[-1]:8.2f} MB")

        self.results['throughput'] = {
            'batch_sizes': batch_sizes,
            'throughputs': throughputs,
            'latencies': latencies,
            'memory_usage': memory_usage
        }

        return self.results['throughput']

    def benchmark_model_sizes(
        self,
        input_dim: int = 1024,
        hidden_dims: List[int] = [128, 256, 512, 1024, 2048],
        batch_size: int = 32,
        num_iterations: int = 100
    ) -> Dict[str, List[float]]:
        """Measure performance across different model sizes"""

        print(f"\n{'='*70}")
        print(f"MODEL SIZE BENCHMARK - {self.device.upper()}")
        print(f"{'='*70}")

        throughputs = []
        param_counts = []
        memory_usage = []

        for hidden_dim in hidden_dims:
            model = UnifiedBrainNetwork(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=4
            ).to(self.device)
            model.eval()

            # Count parameters
            params = sum(p.numel() for p in model.parameters())
            param_counts.append(params / 1e6)  # Millions

            # Benchmark
            x = torch.randn(batch_size, input_dim).to(self.device)

            times = []
            with torch.no_grad():
                for _ in range(num_iterations):
                    start = time.perf_counter()
                    _ = model(x)

                    if self.device == 'cuda':
                        torch.cuda.synchronize()

                    end = time.perf_counter()
                    times.append(end - start)

            avg_time = np.mean(times)
            throughput = batch_size / avg_time
            throughputs.append(throughput)

            # Memory
            if self.device == 'cuda':
                mem = torch.cuda.max_memory_allocated() / 1024**2
                memory_usage.append(mem)
                torch.cuda.reset_peak_memory_stats()
            else:
                memory_usage.append(0)

            print(f"Hidden dim: {hidden_dim:4d} | "
                  f"Params: {param_counts[-1]:6.2f}M | "
                  f"Throughput: {throughput:8.2f} samples/sec | "
                  f"Memory: {memory_usage[-1]:8.2f} MB")

        self.results['model_sizes'] = {
            'hidden_dims': hidden_dims,
            'param_counts': param_counts,
            'throughputs': throughputs,
            'memory_usage': memory_usage
        }

        return self.results['model_sizes']

    def benchmark_precision(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        batch_size: int = 32,
        num_iterations: int = 100
    ) -> Dict[str, Dict]:
        """Compare FP32 vs FP16 performance"""

        print(f"\n{'='*70}")
        print(f"PRECISION BENCHMARK - {self.device.upper()}")
        print(f"{'='*70}")

        precisions = {}

        for dtype_name, dtype in [('FP32', torch.float32), ('FP16', torch.float16)]:
            if dtype == torch.float16 and self.device == 'cpu':
                print(f"Skipping {dtype_name} on CPU (not supported)")
                continue

            model = UnifiedBrainNetwork(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=4
            ).to(self.device).to(dtype)
            model.eval()

            x = torch.randn(batch_size, input_dim).to(self.device).to(dtype)

            times = []
            with torch.no_grad():
                for _ in range(num_iterations):
                    start = time.perf_counter()
                    _ = model(x)

                    if self.device == 'cuda':
                        torch.cuda.synchronize()

                    end = time.perf_counter()
                    times.append(end - start)

            avg_time = np.mean(times)
            throughput = batch_size / avg_time

            if self.device == 'cuda':
                mem = torch.cuda.max_memory_allocated() / 1024**2
                torch.cuda.reset_peak_memory_stats()
            else:
                mem = 0

            precisions[dtype_name] = {
                'throughput': throughput,
                'latency': (avg_time / batch_size) * 1000,
                'memory': mem
            }

            print(f"{dtype_name}: {throughput:8.2f} samples/sec | "
                  f"Memory: {mem:8.2f} MB")

        self.results['precision'] = precisions
        return precisions

    def plot_results(self, save_dir: str = 'benchmark_results'):
        """Generate visualization plots"""

        Path(save_dir).mkdir(exist_ok=True)

        # Plot 1: Throughput vs Batch Size
        if 'throughput' in self.results:
            data = self.results['throughput']

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            # Throughput
            axes[0].plot(data['batch_sizes'], data['throughputs'], 'o-', linewidth=2)
            axes[0].set_xlabel('Batch Size')
            axes[0].set_ylabel('Throughput (samples/sec)')
            axes[0].set_title('Throughput vs Batch Size')
            axes[0].grid(True, alpha=0.3)

            # Latency
            axes[1].plot(data['batch_sizes'], data['latencies'], 's-', linewidth=2, color='orange')
            axes[1].set_xlabel('Batch Size')
            axes[1].set_ylabel('Latency (ms/sample)')
            axes[1].set_title('Latency vs Batch Size')
            axes[1].grid(True, alpha=0.3)

            # Memory
            if self.device == 'cuda':
                axes[2].plot(data['batch_sizes'], data['memory_usage'], '^-', linewidth=2, color='green')
                axes[2].set_xlabel('Batch Size')
                axes[2].set_ylabel('GPU Memory (MB)')
                axes[2].set_title('Memory Usage vs Batch Size')
                axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'{save_dir}/throughput_benchmark.png', dpi=300, bbox_inches='tight')
            print(f"\nSaved throughput plot to {save_dir}/throughput_benchmark.png")

        # Plot 2: Model Size Scaling
        if 'model_sizes' in self.results:
            data = self.results['model_sizes']

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # Throughput vs Parameters
            axes[0].plot(data['param_counts'], data['throughputs'], 'o-', linewidth=2)
            axes[0].set_xlabel('Parameters (Millions)')
            axes[0].set_ylabel('Throughput (samples/sec)')
            axes[0].set_title('Throughput vs Model Size')
            axes[0].grid(True, alpha=0.3)

            # Memory vs Parameters
            if self.device == 'cuda':
                axes[1].plot(data['param_counts'], data['memory_usage'], 's-', linewidth=2, color='red')
                axes[1].set_xlabel('Parameters (Millions)')
                axes[1].set_ylabel('GPU Memory (MB)')
                axes[1].set_title('Memory vs Model Size')
                axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'{save_dir}/model_size_benchmark.png', dpi=300, bbox_inches='tight')
            print(f"Saved model size plot to {save_dir}/model_size_benchmark.png")

        plt.close('all')

    def save_report(self, filepath: str = 'benchmark_results/BENCHMARK_REPORT.md'):
        """Generate markdown report"""

        Path(filepath).parent.mkdir(exist_ok=True)

        with open(filepath, 'w') as f:
            f.write("# NeuralLayers Inference Benchmark Report\n\n")
            f.write(f"**Device**: {self.device.upper()}\n")
            f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Throughput section
            if 'throughput' in self.results:
                f.write("## Throughput vs Batch Size\n\n")
                f.write("| Batch Size | Throughput (samples/sec) | Latency (ms/sample) | Memory (MB) |\n")
                f.write("|------------|--------------------------|---------------------|-------------|\n")

                data = self.results['throughput']
                for i, bs in enumerate(data['batch_sizes']):
                    f.write(f"| {bs:4d} | {data['throughputs'][i]:8.2f} | "
                           f"{data['latencies'][i]:6.2f} | {data['memory_usage'][i]:8.2f} |\n")
                f.write("\n")

            # Model size section
            if 'model_sizes' in self.results:
                f.write("## Model Size Scaling\n\n")
                f.write("| Hidden Dim | Parameters (M) | Throughput (samples/sec) | Memory (MB) |\n")
                f.write("|------------|----------------|--------------------------|-------------|\n")

                data = self.results['model_sizes']
                for i, hd in enumerate(data['hidden_dims']):
                    f.write(f"| {hd:4d} | {data['param_counts'][i]:6.2f} | "
                           f"{data['throughputs'][i]:8.2f} | {data['memory_usage'][i]:8.2f} |\n")
                f.write("\n")

            # Precision section
            if 'precision' in self.results:
                f.write("## Precision Comparison\n\n")
                f.write("| Precision | Throughput (samples/sec) | Latency (ms) | Memory (MB) | Speedup |\n")
                f.write("|-----------|--------------------------|--------------|-------------|----------|\n")

                data = self.results['precision']
                fp32_throughput = data.get('FP32', {}).get('throughput', 1.0)

                for precision, metrics in data.items():
                    speedup = metrics['throughput'] / fp32_throughput
                    f.write(f"| {precision:5s} | {metrics['throughput']:8.2f} | "
                           f"{metrics['latency']:6.2f} | {metrics['memory']:8.2f} | {speedup:.2f}x |\n")

        print(f"\nSaved benchmark report to {filepath}")


def main():
    """Run all benchmarks"""

    print("🚀 NeuralLayers Inference Benchmark Suite")
    print("=" * 70)

    # Initialize benchmark
    benchmark = InferenceBenchmark()

    # Run benchmarks
    benchmark.benchmark_throughput(
        batch_sizes=[1, 4, 8, 16, 32, 64, 128]
    )

    benchmark.benchmark_model_sizes(
        hidden_dims=[128, 256, 512, 1024]
    )

    benchmark.benchmark_precision()

    # Generate visualizations and report
    benchmark.plot_results()
    benchmark.save_report()

    print("\n✅ Benchmark suite completed!")
    print("📊 Results saved to benchmark_results/")


if __name__ == '__main__':
    main()
