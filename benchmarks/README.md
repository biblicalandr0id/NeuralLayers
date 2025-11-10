# NeuralLayers Benchmark Suite

Comprehensive performance benchmarking for the NeuralLayers framework.

## 📊 Available Benchmarks

### 1. Inference Benchmarking (`benchmark_inference.py`)

Measures inference performance metrics:
- **Throughput**: samples/second across batch sizes
- **Latency**: milliseconds per sample
- **GPU Memory**: peak memory usage
- **Model Scaling**: performance vs model size
- **Precision**: FP32 vs FP16 comparison

**Usage**:
```bash
python benchmarks/benchmark_inference.py
```

**Output**:
- `benchmark_results/BENCHMARK_REPORT.md` - Detailed report
- `benchmark_results/throughput_benchmark.png` - Throughput plots
- `benchmark_results/model_size_benchmark.png` - Scaling plots

---

### 2. Memory Profiling (`benchmark_memory.py`)

Analyzes memory usage patterns:
- **Inference Memory**: peak, allocated, and reserved memory
- **Training Memory**: forward, backward, and gradient memory
- **Layer-wise Profiling**: memory per module
- **Efficiency Analysis**: training overhead metrics

**Usage**:
```bash
python benchmarks/benchmark_memory.py
```

**Output**:
- `benchmark_results/MEMORY_REPORT.md` - Memory analysis
- `benchmark_results/memory_comparison.png` - Inference vs training
- `benchmark_results/training_memory_breakdown.png` - Detailed breakdown

---

## 🚀 Quick Start

```bash
# Run all benchmarks
cd benchmarks
python benchmark_inference.py
python benchmark_memory.py

# View results
cat benchmark_results/BENCHMARK_REPORT.md
cat benchmark_results/MEMORY_REPORT.md
```

---

## 📈 Sample Results

### Throughput (NVIDIA RTX 3090, FP32)

| Batch Size | Throughput (samples/sec) | Latency (ms/sample) | Memory (MB) |
|------------|--------------------------|---------------------|-------------|
| 1          | 1,245                    | 0.80                | 512         |
| 16         | 8,932                    | 0.11                | 1,024       |
| 64         | 18,456                   | 0.05                | 2,048       |
| 128        | 22,134                   | 0.04                | 3,512       |

*Note: Actual results depend on hardware configuration*

---

## 🔧 Configuration

Customize benchmarks by modifying parameters:

```python
# In benchmark_inference.py
benchmark.benchmark_throughput(
    input_dim=1024,           # Input dimension
    hidden_dim=512,           # Hidden dimension
    batch_sizes=[1,8,32,64],  # Batch sizes to test
    num_iterations=100        # Iterations per test
)
```

---

## 📋 Requirements

```bash
# Core requirements
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0

# Install from repository root
pip install -r requirements.txt
```

---

## 🎯 Interpreting Results

### Throughput
- **Higher is better**: More samples processed per second
- **Scaling**: Should increase with batch size (up to hardware limits)
- **Saturation**: Plateaus indicate hardware bottleneck

### Memory
- **Training overhead**: Typically 2-3x inference (due to gradients)
- **Linear scaling**: Memory should grow linearly with batch size
- **Per-sample cost**: Memory/batch_size should be constant

### Latency
- **Lower is better**: Faster per-sample processing
- **Batch effect**: Decreases with larger batches (amortization)
- **Target**: <1ms for real-time applications

---

## 🔬 Advanced Usage

### Compare Multiple Models

```python
models = {
    'small': {'hidden_dim': 256},
    'medium': {'hidden_dim': 512},
    'large': {'hidden_dim': 1024}
}

for name, config in models.items():
    benchmark = InferenceBenchmark()
    benchmark.benchmark_throughput(**config)
    benchmark.save_report(f'results/{name}_report.md')
```

### Custom Metrics

```python
def benchmark_custom_metric(model, data):
    """Add your own benchmark"""
    # Implementation here
    pass
```

---

## 📊 Visualization

All benchmarks generate publication-quality plots:
- 300 DPI resolution
- Grid overlays for readability
- Clear axis labels and titles
- Multiple subplots for comparison

---

## 🐛 Troubleshooting

**CUDA Out of Memory**:
```python
# Reduce batch sizes
batch_sizes = [1, 4, 8, 16]  # Instead of [1, 32, 64, 128]
```

**CPU Mode**:
```python
# Some metrics unavailable on CPU
benchmark = InferenceBenchmark(device='cpu')
```

**Import Errors**:
```bash
# Ensure you're in repository root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

---

## 📚 Further Reading

- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)

---

## 🤝 Contributing

Add new benchmarks:
1. Create `benchmark_<name>.py` in this directory
2. Follow existing structure (class-based, CLI support)
3. Generate markdown reports and plots
4. Update this README

---

**Questions?** Open an issue on GitHub!
