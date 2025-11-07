# NeuralLayers Examples

Simple, self-contained examples demonstrating core NeuralLayers functionality.

## 📁 Available Examples

### 1. Simple Network (`simple_network.py`)

**What it does**: Demonstrates basic model initialization and inference

**Run**:
```bash
python examples/simple_network.py
```

**What you'll learn**:
- How to create a UnifiedBrainNetwork
- How to perform a forward pass
- How to access system state (V, NT, Ca, ATP, g, Ψ, τ, ω)
- How to check biological constraints

**Output**:
```
Configuration:
  Input dimension:  1024
  Hidden dimension: 512
  Batch size:       16

Results:
Output                         Shape                Mean         Std
----------------------------------------------------------------------
Final output                   (16, 1024)           0.0234       0.9876
Membrane potential (V)         (16, 512)            -65.23       5.42
...

✅ All biological constraints satisfied!
```

---

### 2. Basic Training (`basic_training.py`)

**What it does**: Trains a model on synthetic data

**Run**:
```bash
python examples/basic_training.py
```

**What you'll learn**:
- How to generate training data
- How to set up a training loop
- How to use PyTorch optimizers
- How to track and visualize loss
- How to save/load checkpoints

**Output**:
- Training progress printed to console
- Loss curves saved to `outputs/training_progress.png`
- Model checkpoint saved to `checkpoints/basic_training.pth`

---

## 🚀 More Advanced Examples

For more complex examples, see:

- **Jupyter Notebooks**: [`../notebooks/`](../notebooks/)
  - 01_getting_started.ipynb
  - 02_brain_dynamics.ipynb
  - 03_consciousness_exploration.ipynb

- **Production Tools**:
  - Full training pipeline: `python train.py`
  - Model export: `python export_model.py`
  - Interactive demo: `python demo_app.py`

- **Research Applications**:
  - Consciousness research: `python applications/consciousness_research.py`
  - Real-time monitoring: `python applications/realtime_monitoring.py`

---

## 🎓 Learning Path

Recommended order for learning:

1. **Start here**: `simple_network.py` - Understand the basics
2. **Next**: `basic_training.py` - Learn to train models
3. **Then**: Jupyter notebooks - Interactive exploration
4. **Advanced**: Production tools and research applications

---

## 🛠️ Requirements

All examples require:
```bash
pip install torch numpy matplotlib
```

Or install from repository root:
```bash
pip install -r requirements.txt
```

---

## 💡 Tips

### Running on GPU

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x = x.to(device)
```

### Adjusting Model Size

```python
# Smaller model (faster, less memory)
model = UnifiedBrainNetwork(
    input_dim=256,
    hidden_dim=128,
    num_layers=2
)

# Larger model (slower, more capacity)
model = UnifiedBrainNetwork(
    input_dim=2048,
    hidden_dim=1024,
    num_layers=6
)
```

### Debugging

Add these lines to see intermediate outputs:

```python
# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# Print shapes
print(f"Input shape: {x.shape}")
print(f"Output shape: {output['output'].shape}")

# Check for NaNs
assert not torch.isnan(output['output']).any(), "NaN detected!"
```

---

## 📚 Documentation

For detailed API documentation, see:
- Main README: [`../README.md`](../README.md)
- Contributing guide: [`../CONTRIBUTING.md`](../CONTRIBUTING.md)
- Quick start: [`../QUICKSTART.md`](../QUICKSTART.md)

---

## 🤝 Contributing

Found a bug? Have an idea for a new example?

1. Open an issue on GitHub
2. Submit a pull request
3. See [`../CONTRIBUTING.md`](../CONTRIBUTING.md) for guidelines

---

## 📞 Need Help?

- 📖 Read the [main documentation](../README.md)
- 💬 Open a [GitHub issue](https://github.com/biblicalandr0id/NeuralLayers/issues)
- 📧 Contact: [repository owner]

---

**Happy learning! 🚀**
