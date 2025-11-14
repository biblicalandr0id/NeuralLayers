"""
Production-Grade Model Optimization for NeuralLayers

Features:
- Quantization (INT8, FP16, dynamic)
- Pruning (structured, unstructured, gradual)
- Knowledge Distillation
- ONNX Export with optimization
- TorchScript export
- Model compression metrics
- Latency profiling
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import quantize_dynamic, QuantStub, DeQuantStub
from torch.nn.utils import prune
import torch.onnx
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import time


@dataclass
class OptimizationConfig:
    """Configuration for model optimization"""

    # Quantization
    quantization_method: str = "dynamic"  # dynamic, static, qat
    quantization_dtype: str = "qint8"  # qint8, float16

    # Pruning
    pruning_method: str = "l1_unstructured"  # l1_unstructured, random_unstructured, ln_structured
    pruning_amount: float = 0.3  # Fraction of parameters to prune
    pruning_schedule: str = "iterative"  # iterative, oneshot
    pruning_iterations: int = 10

    # Knowledge Distillation
    temperature: float = 3.0
    alpha: float = 0.7  # Weight for distillation loss

    # Export
    onnx_opset_version: int = 14
    onnx_optimize: bool = True
    export_dynamic_axes: bool = True

    # Profiling
    profile_iterations: int = 100
    warmup_iterations: int = 10


class QuantizationOptimizer:
    """
    Quantization optimizer for model compression

    Supports:
    - Dynamic quantization (post-training)
    - Static quantization (requires calibration)
    - Quantization-aware training (QAT)
    """

    def __init__(self, config: OptimizationConfig):
        self.config = config

    def dynamic_quantize(
        self,
        model: nn.Module,
        dtype: torch.dtype = torch.qint8
    ) -> nn.Module:
        """Apply dynamic quantization (easiest, no calibration needed)"""

        # Quantize Linear and LSTM layers
        quantized_model = quantize_dynamic(
            model,
            {nn.Linear, nn.LSTM, nn.GRU},
            dtype=dtype
        )

        return quantized_model

    def prepare_static_quantize(
        self,
        model: nn.Module
    ) -> nn.Module:
        """Prepare model for static quantization"""

        model.eval()

        # Specify quantization configuration
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

        # Fuse modules (Conv+BN+ReLU, etc.)
        # Note: NeuralLayers may need custom fusions
        torch.quantization.fuse_modules(model, [['conv', 'relu']], inplace=True)

        # Prepare for calibration
        torch.quantization.prepare(model, inplace=True)

        return model

    def calibrate(
        self,
        model: nn.Module,
        calibration_data: torch.utils.data.DataLoader
    ):
        """Calibrate model on representative data"""

        model.eval()
        with torch.no_grad():
            for batch in calibration_data:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch

                model(inputs)

    def convert_static_quantize(
        self,
        model: nn.Module
    ) -> nn.Module:
        """Convert calibrated model to quantized model"""

        torch.quantization.convert(model, inplace=True)
        return model

    def quantize_to_fp16(
        self,
        model: nn.Module
    ) -> nn.Module:
        """Convert model to FP16"""

        return model.half()

    def measure_model_size(
        self,
        model: nn.Module
    ) -> Dict[str, float]:
        """Measure model size in MB"""

        # Save to temporary file
        tmp_path = "/tmp/model_size_check.pt"
        torch.save(model.state_dict(), tmp_path)

        size_mb = os.path.getsize(tmp_path) / (1024 * 1024)

        os.remove(tmp_path)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            "size_mb": size_mb,
            "total_params": total_params,
            "trainable_params": trainable_params
        }


class PruningOptimizer:
    """
    Neural network pruning for model compression

    Supports:
    - Unstructured pruning (individual weights)
    - Structured pruning (entire channels/neurons)
    - Gradual pruning (iterative)
    """

    def __init__(self, config: OptimizationConfig):
        self.config = config

    def l1_unstructured_prune(
        self,
        model: nn.Module,
        amount: float = 0.3
    ) -> nn.Module:
        """Apply L1 unstructured pruning to all Linear layers"""

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=amount)

        return model

    def random_unstructured_prune(
        self,
        model: nn.Module,
        amount: float = 0.3
    ) -> nn.Module:
        """Apply random unstructured pruning"""

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                prune.random_unstructured(module, name='weight', amount=amount)

        return model

    def ln_structured_prune(
        self,
        model: nn.Module,
        amount: float = 0.3,
        n: int = 2,
        dim: int = 0
    ) -> nn.Module:
        """Apply Ln structured pruning (prune entire dimensions)"""

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                prune.ln_structured(
                    module,
                    name='weight',
                    amount=amount,
                    n=n,
                    dim=dim
                )

        return model

    def iterative_prune(
        self,
        model: nn.Module,
        train_fn: callable,
        val_fn: callable,
        initial_sparsity: float = 0.0,
        final_sparsity: float = 0.5,
        iterations: int = 10
    ) -> nn.Module:
        """
        Iterative magnitude pruning with retraining

        Args:
            model: Model to prune
            train_fn: Function to train model (takes model, returns trained model)
            val_fn: Function to validate model (takes model, returns accuracy)
            initial_sparsity: Starting sparsity
            final_sparsity: Target sparsity
            iterations: Number of pruning iterations
        """

        sparsity_schedule = np.linspace(
            initial_sparsity,
            final_sparsity,
            iterations
        )

        results = []

        for i, target_sparsity in enumerate(sparsity_schedule):
            print(f"Iteration {i+1}/{iterations}: Target sparsity {target_sparsity:.2%}")

            # Prune
            self.l1_unstructured_prune(model, amount=target_sparsity)

            # Retrain
            model = train_fn(model)

            # Evaluate
            accuracy = val_fn(model)

            results.append({
                "iteration": i + 1,
                "sparsity": target_sparsity,
                "accuracy": accuracy
            })

            print(f"  Accuracy: {accuracy:.4f}")

        # Make pruning permanent
        self.remove_pruning_reparameterization(model)

        return model, results

    def remove_pruning_reparameterization(
        self,
        model: nn.Module
    ) -> nn.Module:
        """Remove pruning reparameterization to make permanent"""

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                try:
                    prune.remove(module, 'weight')
                except ValueError:
                    pass  # No pruning applied

        return model

    def measure_sparsity(
        self,
        model: nn.Module
    ) -> Dict[str, float]:
        """Measure global and per-layer sparsity"""

        global_zeros = 0
        global_params = 0

        layer_sparsity = {}

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                zeros = (module.weight == 0).sum().item()
                total = module.weight.numel()

                global_zeros += zeros
                global_params += total

                layer_sparsity[name] = zeros / total

        global_sparsity = global_zeros / global_params if global_params > 0 else 0

        return {
            "global_sparsity": global_sparsity,
            "layer_sparsity": layer_sparsity
        }


class KnowledgeDistillation:
    """
    Knowledge Distillation for model compression

    Train a small "student" model to match a large "teacher" model
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        config: OptimizationConfig
    ):
        self.teacher = teacher
        self.student = student
        self.config = config

        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
        temperature: float = 3.0,
        alpha: float = 0.7
    ) -> torch.Tensor:
        """
        Compute distillation loss

        Loss = alpha * KL(teacher_soft, student_soft) + (1-alpha) * CE(student, targets)
        """

        # Soft targets (with temperature)
        soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / temperature, dim=-1)

        distill_loss = F.kl_div(
            soft_student,
            soft_teacher,
            reduction='batchmean'
        ) * (temperature ** 2)

        # Hard targets
        student_loss = F.cross_entropy(student_logits, targets)

        # Combined loss
        total_loss = alpha * distill_loss + (1 - alpha) * student_loss

        return total_loss

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """Single training step with distillation"""

        # Teacher predictions (no grad)
        with torch.no_grad():
            teacher_output = self.teacher(inputs)
            if isinstance(teacher_output, dict):
                teacher_logits = teacher_output['output']
            else:
                teacher_logits = teacher_output

        # Student predictions
        student_output = self.student(inputs)
        if isinstance(student_output, dict):
            student_logits = student_output['output']
        else:
            student_logits = student_output

        # Compute loss
        loss = self.distillation_loss(
            student_logits,
            teacher_logits,
            targets,
            temperature=self.config.temperature,
            alpha=self.config.alpha
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()


class ModelExporter:
    """
    Export models to various formats

    Supports:
    - ONNX (with optimization)
    - TorchScript
    - TensorRT (via ONNX)
    """

    def __init__(self, config: OptimizationConfig):
        self.config = config

    def export_onnx(
        self,
        model: nn.Module,
        output_path: str,
        input_shape: Tuple[int, ...],
        input_names: List[str] = None,
        output_names: List[str] = None,
        dynamic_axes: Dict[str, Dict[int, str]] = None
    ):
        """Export model to ONNX format"""

        model.eval()

        # Create dummy input
        dummy_input = torch.randn(*input_shape)

        # Default names
        if input_names is None:
            input_names = ['input']
        if output_names is None:
            output_names = ['output']

        # Dynamic axes for variable batch size
        if dynamic_axes is None and self.config.export_dynamic_axes:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }

        # Export
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=self.config.onnx_opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes
        )

        print(f"✅ ONNX model exported to {output_path}")

        # Optimize ONNX model
        if self.config.onnx_optimize:
            self.optimize_onnx(output_path)

    def optimize_onnx(self, onnx_path: str):
        """Optimize ONNX model (requires onnx package)"""

        try:
            import onnx
            from onnx import optimizer

            # Load model
            model = onnx.load(onnx_path)

            # Optimize
            passes = [
                'eliminate_deadend',
                'eliminate_identity',
                'eliminate_nop_dropout',
                'eliminate_nop_pad',
                'eliminate_nop_transpose',
                'eliminate_unused_initializer',
                'extract_constant_to_initializer',
                'fuse_bn_into_conv',
                'fuse_consecutive_squeezes',
                'fuse_consecutive_transposes',
                'fuse_matmul_add_bias_into_gemm',
                'fuse_pad_into_conv',
                'fuse_transpose_into_gemm'
            ]

            optimized_model = optimizer.optimize(model, passes)

            # Save
            onnx.save(optimized_model, onnx_path)

            print(f"✅ ONNX model optimized")

        except ImportError:
            print("⚠️  onnx package not installed, skipping optimization")

    def export_torchscript(
        self,
        model: nn.Module,
        output_path: str,
        input_shape: Tuple[int, ...],
        method: str = "trace"  # trace or script
    ):
        """Export model to TorchScript"""

        model.eval()

        if method == "trace":
            # Trace method (more compatible but less flexible)
            dummy_input = torch.randn(*input_shape)
            traced = torch.jit.trace(model, dummy_input)
            traced.save(output_path)

        elif method == "script":
            # Script method (more flexible but requires annotations)
            scripted = torch.jit.script(model)
            scripted.save(output_path)

        else:
            raise ValueError(f"Unknown method: {method}")

        print(f"✅ TorchScript model exported to {output_path}")

    def verify_onnx_export(
        self,
        onnx_path: str,
        pytorch_model: nn.Module,
        input_shape: Tuple[int, ...]
    ) -> bool:
        """Verify ONNX export matches PyTorch model"""

        try:
            import onnxruntime as ort

            # Create test input
            test_input = torch.randn(*input_shape)

            # PyTorch inference
            pytorch_model.eval()
            with torch.no_grad():
                pytorch_output = pytorch_model(test_input)
                if isinstance(pytorch_output, dict):
                    pytorch_output = pytorch_output['output']
                pytorch_output = pytorch_output.numpy()

            # ONNX inference
            ort_session = ort.InferenceSession(onnx_path)
            ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
            ort_output = ort_session.run(None, ort_inputs)[0]

            # Compare
            max_diff = np.abs(pytorch_output - ort_output).max()

            if max_diff < 1e-5:
                print(f"✅ ONNX export verified (max diff: {max_diff:.2e})")
                return True
            else:
                print(f"⚠️  ONNX export may have issues (max diff: {max_diff:.2e})")
                return False

        except ImportError:
            print("⚠️  onnxruntime not installed, skipping verification")
            return False


class LatencyProfiler:
    """Profile model latency and throughput"""

    def __init__(self, config: OptimizationConfig):
        self.config = config

    def profile_inference(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        device: str = "cpu"
    ) -> Dict[str, float]:
        """Profile inference latency and throughput"""

        model.eval()
        model.to(device)

        # Warmup
        dummy_input = torch.randn(*input_shape, device=device)
        with torch.no_grad():
            for _ in range(self.config.warmup_iterations):
                model(dummy_input)

        # Benchmark
        latencies = []

        with torch.no_grad():
            for _ in range(self.config.profile_iterations):
                start = time.perf_counter()
                output = model(dummy_input)

                # Synchronize if GPU
                if device != "cpu":
                    torch.cuda.synchronize()

                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # Convert to ms

        latencies = np.array(latencies)

        batch_size = input_shape[0]

        return {
            "mean_latency_ms": latencies.mean(),
            "std_latency_ms": latencies.std(),
            "min_latency_ms": latencies.min(),
            "max_latency_ms": latencies.max(),
            "p50_latency_ms": np.percentile(latencies, 50),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "throughput_samples_per_sec": 1000 * batch_size / latencies.mean(),
            "batch_size": batch_size
        }

    def compare_models(
        self,
        models: Dict[str, nn.Module],
        input_shape: Tuple[int, ...],
        device: str = "cpu"
    ):
        """Compare latency of multiple models"""

        results = {}

        for name, model in models.items():
            print(f"Profiling {name}...")
            results[name] = self.profile_inference(model, input_shape, device)

        # Print comparison
        print("\n" + "="*80)
        print("LATENCY COMPARISON")
        print("="*80)

        for name, metrics in results.items():
            print(f"\n{name}:")
            print(f"  Mean latency:  {metrics['mean_latency_ms']:.2f} ms")
            print(f"  P95 latency:   {metrics['p95_latency_ms']:.2f} ms")
            print(f"  Throughput:    {metrics['throughput_samples_per_sec']:.1f} samples/sec")

        return results


# Example usage
if __name__ == "__main__":
    from logicalbrain_network import UnifiedBrainLogicNetwork

    # Create model
    model = UnifiedBrainLogicNetwork(
        input_dim=128,
        hidden_dim=128,
        output_dim=64
    )

    config = OptimizationConfig()

    print("="*80)
    print("MODEL OPTIMIZATION DEMO")
    print("="*80)

    # 1. Measure baseline
    quant_opt = QuantizationOptimizer(config)
    baseline_size = quant_opt.measure_model_size(model)
    print(f"\n📊 Baseline Model:")
    print(f"  Size: {baseline_size['size_mb']:.2f} MB")
    print(f"  Parameters: {baseline_size['total_params']:,}")

    # 2. Dynamic quantization
    print(f"\n🔧 Applying dynamic quantization...")
    quantized_model = quant_opt.dynamic_quantize(model.cpu())
    quantized_size = quant_opt.measure_model_size(quantized_model)
    print(f"  Size: {quantized_size['size_mb']:.2f} MB")
    print(f"  Compression: {baseline_size['size_mb'] / quantized_size['size_mb']:.2f}x")

    # 3. Pruning
    print(f"\n✂️  Applying L1 unstructured pruning (30%)...")
    prune_opt = PruningOptimizer(config)
    pruned_model = prune_opt.l1_unstructured_prune(
        UnifiedBrainLogicNetwork(input_dim=128, hidden_dim=128, output_dim=64),
        amount=0.3
    )
    sparsity = prune_opt.measure_sparsity(pruned_model)
    print(f"  Global sparsity: {sparsity['global_sparsity']:.1%}")

    # 4. Export to ONNX
    print(f"\n📦 Exporting to ONNX...")
    exporter = ModelExporter(config)
    os.makedirs("exports", exist_ok=True)
    exporter.export_onnx(
        model,
        "exports/model.onnx",
        input_shape=(1, 128)
    )

    # 5. Export to TorchScript
    print(f"\n📦 Exporting to TorchScript...")
    exporter.export_torchscript(
        model,
        "exports/model.pt",
        input_shape=(1, 128),
        method="trace"
    )

    # 6. Profile latency
    print(f"\n⏱️  Profiling latency...")
    profiler = LatencyProfiler(config)

    models = {
        "Baseline": model,
        "Quantized": quantized_model
    }

    profiler.compare_models(models, input_shape=(1, 128), device="cpu")

    print("\n✅ Optimization demo complete!")
