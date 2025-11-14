"""
Integration Tests for SOTA Components

Tests:
- SOTA training pipeline end-to-end
- Data pipeline with augmentation
- Model optimization (quantization, pruning, export)
- Serving API
- Monitoring and experiment tracking
- Model enhancements
- Batch inference
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import os
import tempfile
from pathlib import Path
import json


# Import SOTA components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from logicalbrain_network import UnifiedBrainLogicNetwork
from sota_training import TrainingConfig, SOTATrainer
from data_pipeline import DataConfig, DataPipeline, NeuralLayersDataset, create_synthetic_dataset
from model_optimization import (
    OptimizationConfig,
    QuantizationOptimizer,
    PruningOptimizer,
    ModelExporter,
    LatencyProfiler
)
from model_enhancements import (
    FlashAttention,
    GroupedQueryAttention,
    RotaryPositionalEmbedding,
    SwiGLU,
    RMSNorm,
    EnhancedTransformerBlock
)


class TestSOTATraining:
    """Test SOTA training pipeline"""

    def test_basic_training_loop(self):
        """Test that training loop runs without errors"""

        # Create model
        model = UnifiedBrainLogicNetwork(
            input_dim=128,
            hidden_dim=128,
            output_dim=64
        )

        # Create synthetic data
        X_train, y_train = create_synthetic_dataset(num_samples=100, input_dim=128, output_dim=64)
        X_val, y_val = create_synthetic_dataset(num_samples=20, input_dim=128, output_dim=64)

        train_dataset = NeuralLayersDataset(X_train, y_train)
        val_dataset = NeuralLayersDataset(X_val, y_val)

        # Training config
        config = TrainingConfig(
            epochs=2,
            batch_size=16,
            learning_rate=1e-4,
            use_amp=False,  # Disable AMP for CPU
            early_stopping=False
        )

        # Create trainer
        trainer = SOTATrainer(model, config)

        # Train
        history = trainer.train(train_dataset, val_dataset)

        # Assertions
        assert len(history['train_loss']) == 2
        assert len(history['val_loss']) == 2
        assert all(isinstance(x, float) for x in history['train_loss'])
        assert all(isinstance(x, float) for x in history['val_loss'])

    def test_mixed_precision_training(self):
        """Test mixed precision training (if CUDA available)"""

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model = UnifiedBrainLogicNetwork(128, 128, 64).cuda()

        X_train, y_train = create_synthetic_dataset(num_samples=100, input_dim=128, output_dim=64)
        train_dataset = NeuralLayersDataset(X_train, y_train)

        config = TrainingConfig(
            epochs=1,
            batch_size=16,
            use_amp=True,
            device="cuda"
        )

        trainer = SOTATrainer(model, config)
        history = trainer.train(train_dataset, train_dataset)

        assert len(history['train_loss']) == 1

    def test_gradient_accumulation(self):
        """Test gradient accumulation"""

        model = UnifiedBrainLogicNetwork(128, 128, 64)

        X_train, y_train = create_synthetic_dataset(num_samples=100, input_dim=128, output_dim=64)
        train_dataset = NeuralLayersDataset(X_train, y_train)

        config = TrainingConfig(
            epochs=1,
            batch_size=8,
            gradient_accumulation_steps=4,
            use_amp=False
        )

        trainer = SOTATrainer(model, config)
        history = trainer.train(train_dataset, train_dataset)

        assert len(history['train_loss']) == 1

    def test_checkpoint_saving(self):
        """Test checkpoint saving and loading"""

        with tempfile.TemporaryDirectory() as tmpdir:
            model = UnifiedBrainLogicNetwork(128, 128, 64)

            config = TrainingConfig(
                epochs=2,
                checkpoint_dir=tmpdir,
                save_every=1
            )

            trainer = SOTATrainer(model, config)

            # Save checkpoint
            trainer.save_checkpoint(epoch=0, metrics={'loss': 0.5}, is_best=True)

            # Check files exist
            checkpoint_files = list(Path(tmpdir).glob("*.pt"))
            assert len(checkpoint_files) > 0

            # Load checkpoint
            checkpoint = torch.load(checkpoint_files[0])
            assert 'epoch' in checkpoint
            assert 'model_state_dict' in checkpoint
            assert 'optimizer_state_dict' in checkpoint


class TestDataPipeline:
    """Test data pipeline"""

    def test_dataset_loading(self):
        """Test dataset can load different data types"""

        # NumPy array
        data = np.random.randn(100, 128).astype(np.float32)
        dataset = NeuralLayersDataset(data)
        assert len(dataset) == 100

        sample = dataset[0]
        assert isinstance(sample, torch.Tensor)
        assert sample.shape == (128,)

    def test_dataset_with_targets(self):
        """Test dataset with targets"""

        X = np.random.randn(100, 128).astype(np.float32)
        y = np.random.randn(100, 64).astype(np.float32)

        dataset = NeuralLayersDataset(X, y)

        data, target = dataset[0]
        assert data.shape == (128,)
        assert target.shape == (64,)

    def test_data_augmentation(self):
        """Test data augmentation"""

        from data_pipeline import RandomNoise, RandomScale, Compose

        X = np.random.randn(100, 128).astype(np.float32)

        # Create transform pipeline
        transform = Compose([
            RandomNoise(std=0.01),
            RandomScale(scale_range=(0.95, 1.05))
        ])

        dataset = NeuralLayersDataset(X, transform=transform)

        sample1 = dataset[0]
        sample2 = dataset[0]

        # Due to randomness, samples should differ
        assert not torch.allclose(sample1, sample2, atol=0.001)

    def test_dataloader_creation(self):
        """Test DataLoader creation"""

        X, y = create_synthetic_dataset(num_samples=100, input_dim=128, output_dim=64)
        dataset = NeuralLayersDataset(X, y)

        config = DataConfig(batch_size=32, num_workers=0)  # 0 workers for testing
        pipeline = DataPipeline(config)

        loader = pipeline.create_dataloader(dataset)

        batch_count = 0
        for batch_x, batch_y in loader:
            batch_count += 1
            assert batch_x.shape[0] <= 32
            assert batch_y.shape[0] <= 32

        assert batch_count > 0


class TestModelOptimization:
    """Test model optimization"""

    def test_dynamic_quantization(self):
        """Test dynamic quantization"""

        model = UnifiedBrainLogicNetwork(128, 128, 64)

        config = OptimizationConfig()
        quantizer = QuantizationOptimizer(config)

        # Get baseline size
        baseline_size = quantizer.measure_model_size(model)

        # Quantize
        quantized_model = quantizer.dynamic_quantize(model)

        # Get quantized size
        quantized_size = quantizer.measure_model_size(quantized_model)

        # Quantized model should be smaller
        assert quantized_size['size_mb'] < baseline_size['size_mb']

        # Model should still work
        x = torch.randn(2, 128)
        output = quantized_model(x)
        assert 'output' in output

    def test_pruning(self):
        """Test model pruning"""

        model = UnifiedBrainLogicNetwork(128, 128, 64)

        config = OptimizationConfig(pruning_amount=0.3)
        pruner = PruningOptimizer(config)

        # Prune
        pruned_model = pruner.l1_unstructured_prune(model, amount=0.3)

        # Measure sparsity
        sparsity = pruner.measure_sparsity(pruned_model)

        # Should have some sparsity
        assert sparsity['global_sparsity'] > 0.0

        # Model should still work
        x = torch.randn(2, 128)
        output = pruned_model(x)
        assert 'output' in output

    def test_onnx_export(self):
        """Test ONNX export"""

        with tempfile.TemporaryDirectory() as tmpdir:
            model = UnifiedBrainLogicNetwork(128, 128, 64)

            config = OptimizationConfig()
            exporter = ModelExporter(config)

            output_path = str(Path(tmpdir) / "model.onnx")

            # Export
            exporter.export_onnx(
                model,
                output_path,
                input_shape=(1, 128)
            )

            # Check file exists
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0

    def test_torchscript_export(self):
        """Test TorchScript export"""

        with tempfile.TemporaryDirectory() as tmpdir:
            model = UnifiedBrainLogicNetwork(128, 128, 64)

            config = OptimizationConfig()
            exporter = ModelExporter(config)

            output_path = str(Path(tmpdir) / "model.pt")

            # Export
            exporter.export_torchscript(
                model,
                output_path,
                input_shape=(1, 128),
                method="trace"
            )

            # Check file exists
            assert os.path.exists(output_path)

            # Load and test
            loaded_model = torch.jit.load(output_path)
            x = torch.randn(1, 128)
            output = loaded_model(x)
            assert output is not None

    def test_latency_profiling(self):
        """Test latency profiling"""

        model = UnifiedBrainLogicNetwork(128, 128, 64)

        config = OptimizationConfig(profile_iterations=10, warmup_iterations=2)
        profiler = LatencyProfiler(config)

        metrics = profiler.profile_inference(model, input_shape=(1, 128), device="cpu")

        # Check all metrics are present
        assert 'mean_latency_ms' in metrics
        assert 'std_latency_ms' in metrics
        assert 'throughput_samples_per_sec' in metrics
        assert metrics['mean_latency_ms'] > 0


class TestModelEnhancements:
    """Test model enhancements"""

    def test_flash_attention(self):
        """Test Flash Attention"""

        dim = 256
        seq_len = 16
        batch_size = 2

        attn = FlashAttention(dim, num_heads=8)

        x = torch.randn(batch_size, seq_len, dim)
        output = attn(x)

        assert output.shape == (batch_size, seq_len, dim)

    def test_grouped_query_attention(self):
        """Test Grouped Query Attention"""

        dim = 256
        seq_len = 16
        batch_size = 2

        gqa = GroupedQueryAttention(dim, num_heads=8, num_kv_heads=2)

        x = torch.randn(batch_size, seq_len, dim)
        output = gqa(x)

        assert output.shape == (batch_size, seq_len, dim)

    def test_rotary_embedding(self):
        """Test Rotary Position Embedding"""

        dim = 64
        seq_len = 16
        batch_size = 2
        num_heads = 8

        rope = RotaryPositionalEmbedding(dim // num_heads)

        q = torch.randn(batch_size, num_heads, seq_len, dim // num_heads)
        k = torch.randn(batch_size, num_heads, seq_len, dim // num_heads)

        q_rot, k_rot = rope(q, k)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_swiglu_activation(self):
        """Test SwiGLU activation"""

        dim = 256
        batch_size = 2
        seq_len = 16

        swiglu = SwiGLU(dim)

        x = torch.randn(batch_size, seq_len, dim)
        output = swiglu(x)

        assert output.shape == (batch_size, seq_len, dim)

    def test_rmsnorm(self):
        """Test RMSNorm"""

        dim = 256
        batch_size = 2
        seq_len = 16

        rmsnorm = RMSNorm(dim)

        x = torch.randn(batch_size, seq_len, dim)
        output = rmsnorm(x)

        assert output.shape == (batch_size, seq_len, dim)

        # Check normalization (RMS should be close to 1)
        rms = torch.sqrt((output ** 2).mean(dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)

    def test_enhanced_transformer_block(self):
        """Test Enhanced Transformer Block"""

        dim = 256
        batch_size = 2
        seq_len = 16

        block = EnhancedTransformerBlock(dim, num_heads=8)

        x = torch.randn(batch_size, seq_len, dim)
        output = block(x)

        assert output.shape == (batch_size, seq_len, dim)

    def test_enhanced_block_gradient_flow(self):
        """Test gradient flow through enhanced block"""

        dim = 256
        batch_size = 2
        seq_len = 16

        block = EnhancedTransformerBlock(dim, num_heads=8)

        x = torch.randn(batch_size, seq_len, dim, requires_grad=True)
        output = block(x)
        loss = output.sum()

        loss.backward()

        # Check gradients exist and are not NaN
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestBatchInference:
    """Test batch inference engine"""

    def test_batch_inference_basic(self):
        """Test basic batch inference"""

        from serving.batch_inference import (
            BatchInferenceConfig,
            BatchInferenceEngine,
            InferenceDataset
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create model
            model = UnifiedBrainLogicNetwork(128, 128, 64)

            # Create dataset
            data = np.random.randn(50, 128).astype(np.float32)
            dataset = InferenceDataset(data)

            # Configure inference
            config = BatchInferenceConfig(
                batch_size=16,
                num_workers=0,
                output_dir=tmpdir,
                save_format="npz",
                checkpoint_interval=10
            )

            # Run inference
            engine = BatchInferenceEngine(model, config)
            results = engine.run(dataset, resume=False)

            # Check results
            assert len(results) == 50
            assert all(isinstance(v, np.ndarray) for v in results.values())

            # Check output files exist
            output_files = list(Path(tmpdir).glob("*.npz"))
            assert len(output_files) > 0


class TestEndToEnd:
    """End-to-end integration tests"""

    def test_complete_training_pipeline(self):
        """Test complete training pipeline from data to trained model"""

        # 1. Create data
        X_train, y_train = create_synthetic_dataset(num_samples=100, input_dim=128, output_dim=64)
        X_val, y_val = create_synthetic_dataset(num_samples=20, input_dim=128, output_dim=64)

        # 2. Create datasets with augmentation
        data_config = DataConfig(batch_size=16, use_augmentation=True)
        pipeline = DataPipeline(data_config)

        train_dataset = NeuralLayersDataset(
            X_train,
            y_train,
            transform=pipeline.create_transforms(train=True)
        )

        val_dataset = NeuralLayersDataset(
            X_val,
            y_val,
            transform=pipeline.create_transforms(train=False)
        )

        # 3. Create model
        model = UnifiedBrainLogicNetwork(128, 128, 64)

        # 4. Train
        train_config = TrainingConfig(
            epochs=2,
            batch_size=16,
            learning_rate=1e-4,
            use_amp=False,
            early_stopping=False
        )

        trainer = SOTATrainer(model, train_config)
        history = trainer.train(train_dataset, val_dataset)

        # 5. Validate
        assert len(history['train_loss']) == 2
        assert len(history['val_loss']) == 2

        # 6. Export model
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ModelExporter(OptimizationConfig())
            output_path = str(Path(tmpdir) / "model.onnx")

            exporter.export_onnx(model, output_path, input_shape=(1, 128))

            assert os.path.exists(output_path)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
