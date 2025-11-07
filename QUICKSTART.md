# 🚀 NeuralLayers Quick Start Guide

Get up and running with NeuralLayers in 5 minutes!

## ⚡ Installation

```bash
# Clone the repository
git clone https://github.com/biblicalandr0id/NeuralLayers.git
cd NeuralLayers

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Your First Neural-Logical Network

```python
import torch
from logicalbrain_network import UnifiedBrainLogicNetwork

# Initialize model
model = UnifiedBrainLogicNetwork(
    input_dim=1024,
    hidden_dim=2048,
    output_dim=512
)

# Create input
x = torch.randn(8, 1024)  # batch_size=8

# Forward pass
output = model(x)

# Access results
final_output = output['output']              # Network output
system_state = output['system_state']        # 8D state vector
membrane_potential = output['membrane_potential']  # V(t)
truth_values = output['truth_values']        # τ(t)

print(f"Output shape: {final_output.shape}")
print(f"ATP level: {system_state['ATP'].mean():.2f} μM")
```

## 📚 Examples

### Brain Simulation

```python
from brain_network_implementation import BrainNetwork

brain = BrainNetwork()

# Multi-modal sensory input (6 modalities)
sensory_input = torch.rand(1, 6)

# Process
outputs, state = brain(sensory_input)

print(f"Membrane potential: {outputs['membrane_potential'].item():.2f} mV")
print(f"ATP: {outputs['ATP'].item():.2f} μM")
```

### Consciousness Modeling

```python
from consciousness_layers import ConsciousnessEmergence

consciousness = ConsciousnessEmergence((7, 7, 7), num_layers=7)

# Initialize quantum state
quantum_state = consciousness.initialize_quantum_state()

# Process a moment
input_moment = torch.randn(7, 7, 7).to(torch.complex64)
conscious_output = consciousness.process_moment(input_moment)

# Analyze layers
for i, layer in enumerate(conscious_output):
    magnitude = torch.abs(layer).mean()
    print(f"Layer {i}: magnitude = {magnitude:.4f}")
```

### Logical Reasoning

```python
from LogicalReasoningLayer import LogicalReasoningLayer

reasoning = LogicalReasoningLayer(
    input_dim=128,
    hidden_dim=256,
    num_premises=3
)

# Create logical premises
premises = torch.randn(1, 3, 128)

# Infer conclusion
conclusion = reasoning(premises)

print(f"Truth value: {conclusion.item():.3f}")  # [-1, 1]
```

### UMI Anomaly Detection

```python
from umi_layer import UMI_Network

umi_network = UMI_Network()

# Monitoring metrics [DeltaR, T, V, A]
data = torch.tensor([
    [0.1, 0.05, 0.2, -0.5],  # Normal
    [1.0, 1.0, 1.0, 1.0]     # Critical
])

# Compute UMI scores with alerts
umi_scores, alerts = umi_network(data, return_alerts=True)

alert_names = ['NORMAL', 'WARNING', 'CRITICAL']
for score, alert in zip(umi_scores, alerts):
    print(f"UMI: {score.item():.4f} → {alert_names[alert.item()]}")
```

## 🔧 Configuration

Use YAML configuration for easy customization:

```python
from utils import Config, DeviceManager

# Load configuration
config = Config('config.yaml')

# Get values
input_dim = config.get('model.input_dim')
learning_rate = config.get('training.learning_rate')

# Setup device (CPU/GPU)
device_manager = DeviceManager(config)
device = device_manager.device

model = model.to(device)
```

## 📊 Utilities

### Logging

```python
from utils import Logger

logger = Logger(config=config)
logger.info("Training started")
logger.warning("Low ATP levels detected")
```

### Checkpointing

```python
from utils import CheckpointManager

checkpoint_mgr = CheckpointManager(save_dir='./checkpoints')

# Save
checkpoint_mgr.save(model, optimizer, epoch=10, metrics={'loss': 0.15})

# Load
checkpoint = checkpoint_mgr.load('checkpoint.pth', model, optimizer)
```

### Visualization

```python
from utils import StateVisualizer

visualizer = StateVisualizer(save_dir='./plots')

# Plot system state
visualizer.plot_system_state(output['system_state'], step=100)
```

### Profiling

```python
from utils import Profiler

profiler = Profiler()

profiler.start("forward_pass")
output = model(x)
profiler.end("forward_pass")

print(profiler.report())
```

## 🧪 Testing

Run the test suite:

```bash
# All tests
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/test_unified_network.py -v

# With coverage
python -m pytest tests/ --cov=. --cov-report=html
```

## 📓 Jupyter Notebooks

Interactive tutorials in `notebooks/`:

1. **01_getting_started.ipynb** - Basic usage and setup
2. **02_brain_dynamics.ipynb** - Brain network simulation
3. **03_consciousness_exploration.ipynb** - Consciousness modeling

Start Jupyter:

```bash
pip install jupyter
jupyter notebook notebooks/
```

## 🎮 Interactive Demo

### Command Line

```bash
python demo_app.py
```

### Web Interface

```bash
pip install streamlit
streamlit run demo_app.py
```

Then open http://localhost:8501 in your browser.

## 📖 Complete Tutorial

Run the comprehensive tutorial:

```bash
python examples/complete_tutorial.py
```

This demonstrates:
- Configuration loading
- Model initialization
- Training loop
- Checkpointing
- Visualization
- All major components

## 🐛 Troubleshooting

### Import Errors

```bash
# Ensure you're in the project directory
cd NeuralLayers

# Add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### CUDA Issues

```python
# Check CUDA availability
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# Use CPU if needed
device = torch.device('cpu')
model = model.to(device)
```

### Memory Errors

```python
# Reduce batch size
batch_size = 4  # Instead of 32

# Use smaller model
model = UnifiedBrainLogicNetwork(
    input_dim=512,    # Smaller
    hidden_dim=1024,  # Smaller
    output_dim=256    # Smaller
)
```

## 📚 Next Steps

- Read the [full README](README.md) for detailed documentation
- Explore [examples](examples/) for advanced use cases
- Check [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
- Review [tests](tests/) to see usage patterns

## 💡 Tips

1. **Start small**: Begin with small dimensions and batch sizes
2. **Monitor state**: Watch membrane potential and ATP levels
3. **Use config**: Centralize settings in config.yaml
4. **Profile code**: Use the Profiler to find bottlenecks
5. **GPU acceleration**: Use CUDA for faster training

## 🆘 Getting Help

- Check the [README](README.md) for detailed docs
- Review the [examples](examples/) directory
- Look at [test files](tests/) for usage patterns
- Open an issue on GitHub

---

**Happy neural-logical computing!** 🧠⚡

*Get started in 3 lines of code:*
```python
from logicalbrain_network import UnifiedBrainLogicNetwork
model = UnifiedBrainLogicNetwork()
output = model(torch.randn(8, 1024))
```
