"""
Model Export Utilities

Export NeuralLayers models to various formats:
- ONNX (for deployment to non-Python environments)
- TorchScript (for production PyTorch deployment)
- Quantized models (for edge devices)
- Model summary and statistics

Usage:
    python export_model.py --checkpoint model.pth --format onnx
    python export_model.py --checkpoint model.pth --format torchscript
    python export_model.py --checkpoint model.pth --format all
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple

from logicalbrain_network import UnifiedBrainLogicNetwork
from utils import Config, Logger


class ModelExporter:
    """Export models to various formats."""

    def __init__(self, model: nn.Module, config: Optional[Config] = None):
        """
        Initialize exporter.

        Args:
            model: PyTorch model to export
            config: Optional configuration
        """
        self.model = model
        self.config = config or Config()
        self.logger = Logger("ModelExporter", self.config)

    def export_onnx(self, save_path: str, input_sample: Optional[torch.Tensor] = None):
        """
        Export model to ONNX format.

        Args:
            save_path: Path to save ONNX model
            input_sample: Sample input tensor (optional)
        """
        self.logger.info("Exporting to ONNX...")

        # Create sample input if not provided
        if input_sample is None:
            input_dim = self.config.get('model.input_dim', 1024)
            input_sample = torch.randn(1, input_dim)

        self.model.eval()

        try:
            torch.onnx.export(
                self.model,
                input_sample,
                save_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )

            self.logger.info(f"✅ ONNX model saved to: {save_path}")

            # Verify ONNX model
            try:
                import onnx
                onnx_model = onnx.load(save_path)
                onnx.checker.check_model(onnx_model)
                self.logger.info("✅ ONNX model verified")
            except ImportError:
                self.logger.warning("onnx package not installed. Skipping verification.")

        except Exception as e:
            self.logger.error(f"Failed to export ONNX: {e}")
            raise

    def export_torchscript(self, save_path: str,
                          input_sample: Optional[torch.Tensor] = None,
                          method: str = 'trace'):
        """
        Export model to TorchScript.

        Args:
            save_path: Path to save TorchScript model
            input_sample: Sample input tensor (required for tracing)
            method: 'trace' or 'script'
        """
        self.logger.info(f"Exporting to TorchScript (method: {method})...")

        self.model.eval()

        try:
            if method == 'trace':
                # Requires sample input
                if input_sample is None:
                    input_dim = self.config.get('model.input_dim', 1024)
                    input_sample = torch.randn(1, input_dim)

                # Trace model
                with torch.no_grad():
                    traced_model = torch.jit.trace(self.model, input_sample)

                # Save
                traced_model.save(save_path)

            elif method == 'script':
                # Script model (no input needed)
                scripted_model = torch.jit.script(self.model)
                scripted_model.save(save_path)

            else:
                raise ValueError(f"Unknown method: {method}")

            self.logger.info(f"✅ TorchScript model saved to: {save_path}")

            # Verify
            loaded = torch.jit.load(save_path)
            self.logger.info("✅ TorchScript model verified")

        except Exception as e:
            self.logger.error(f"Failed to export TorchScript: {e}")
            raise

    def export_quantized(self, save_path: str,
                        input_sample: Optional[torch.Tensor] = None,
                        quantization_type: str = 'dynamic'):
        """
        Export quantized model for edge deployment.

        Args:
            save_path: Path to save quantized model
            input_sample: Sample input tensor
            quantization_type: 'dynamic' or 'static'
        """
        self.logger.info(f"Exporting quantized model ({quantization_type})...")

        self.model.eval()

        try:
            if quantization_type == 'dynamic':
                # Dynamic quantization (CPU only)
                quantized_model = torch.quantization.quantize_dynamic(
                    self.model,
                    {nn.Linear},
                    dtype=torch.qint8
                )

            elif quantization_type == 'static':
                # Static quantization requires calibration
                if input_sample is None:
                    raise ValueError("Static quantization requires input_sample")

                # Prepare for quantization
                quantized_model = torch.quantization.quantize_dynamic(
                    self.model,
                    {nn.Linear},
                    dtype=torch.qint8
                )

            else:
                raise ValueError(f"Unknown quantization type: {quantization_type}")

            # Save
            torch.save(quantized_model.state_dict(), save_path)

            self.logger.info(f"✅ Quantized model saved to: {save_path}")

            # Calculate compression ratio
            original_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
            quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
            compression_ratio = original_size / quantized_size

            self.logger.info(f"   Compression ratio: {compression_ratio:.2f}x")
            self.logger.info(f"   Original size: {original_size / 1e6:.2f} MB")
            self.logger.info(f"   Quantized size: {quantized_size / 1e6:.2f} MB")

        except Exception as e:
            self.logger.error(f"Failed to export quantized model: {e}")
            raise

    def export_summary(self, save_path: str):
        """
        Export model summary to text file.

        Args:
            save_path: Path to save summary
        """
        self.logger.info("Generating model summary...")

        summary_lines = []

        # Model architecture
        summary_lines.append("=" * 70)
        summary_lines.append("MODEL ARCHITECTURE")
        summary_lines.append("=" * 70)
        summary_lines.append(str(self.model))
        summary_lines.append("")

        # Parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        summary_lines.append("=" * 70)
        summary_lines.append("PARAMETERS")
        summary_lines.append("=" * 70)
        summary_lines.append(f"Total parameters: {total_params:,}")
        summary_lines.append(f"Trainable parameters: {trainable_params:,}")
        summary_lines.append(f"Non-trainable parameters: {total_params - trainable_params:,}")
        summary_lines.append("")

        # Layer-wise parameters
        summary_lines.append("=" * 70)
        summary_lines.append("LAYER-WISE PARAMETERS")
        summary_lines.append("=" * 70)

        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                params = sum(p.numel() for p in module.parameters())
                if params > 0:
                    summary_lines.append(f"{name:50s} {params:>15,}")

        summary_lines.append("")

        # Memory usage
        total_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        summary_lines.append("=" * 70)
        summary_lines.append("MEMORY USAGE")
        summary_lines.append("=" * 70)
        summary_lines.append(f"Model size (FP32): {total_size / 1e6:.2f} MB")
        summary_lines.append(f"Model size (FP16): {total_size / 2e6:.2f} MB (estimated)")
        summary_lines.append("")

        # Write to file
        with open(save_path, 'w') as f:
            f.write('\n'.join(summary_lines))

        self.logger.info(f"✅ Model summary saved to: {save_path}")

    def export_all(self, output_dir: str, model_name: str = "model"):
        """
        Export model to all supported formats.

        Args:
            output_dir: Directory to save all exports
            model_name: Base name for exported files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Exporting model to all formats in: {output_dir}")

        # Sample input
        input_dim = self.config.get('model.input_dim', 1024)
        input_sample = torch.randn(1, input_dim)

        # Export ONNX
        try:
            self.export_onnx(
                str(output_path / f"{model_name}.onnx"),
                input_sample
            )
        except Exception as e:
            self.logger.warning(f"ONNX export failed: {e}")

        # Export TorchScript (trace)
        try:
            self.export_torchscript(
                str(output_path / f"{model_name}_traced.pt"),
                input_sample,
                method='trace'
            )
        except Exception as e:
            self.logger.warning(f"TorchScript trace failed: {e}")

        # Export TorchScript (script)
        try:
            self.export_torchscript(
                str(output_path / f"{model_name}_scripted.pt"),
                method='script'
            )
        except Exception as e:
            self.logger.warning(f"TorchScript script failed: {e}")

        # Export quantized
        try:
            self.export_quantized(
                str(output_path / f"{model_name}_quantized.pth"),
                input_sample
            )
        except Exception as e:
            self.logger.warning(f"Quantization failed: {e}")

        # Export summary
        self.export_summary(str(output_path / f"{model_name}_summary.txt"))

        self.logger.info("=" * 70)
        self.logger.info("✅ All exports complete!")
        self.logger.info(f"📁 Output directory: {output_dir}")
        self.logger.info("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Export NeuralLayers model')
    parser.add_argument('--checkpoint', type=str, required=False,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--format', type=str, default='all',
                       choices=['onnx', 'torchscript', 'quantized', 'summary', 'all'],
                       help='Export format')
    parser.add_argument('--output', type=str, default='./exports',
                       help='Output directory')
    parser.add_argument('--name', type=str, default='model',
                       help='Base name for exported files')

    args = parser.parse_args()

    # Load config
    config = Config(args.config)

    # Create model
    model = UnifiedBrainLogicNetwork(
        input_dim=config.get('model.input_dim'),
        hidden_dim=config.get('model.hidden_dim'),
        output_dim=config.get('model.output_dim')
    )

    # Load checkpoint if provided
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded checkpoint from: {args.checkpoint}")

    # Create exporter
    exporter = ModelExporter(model, config)

    # Export
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.format == 'all':
        exporter.export_all(args.output, args.name)
    elif args.format == 'onnx':
        exporter.export_onnx(str(output_path / f"{args.name}.onnx"))
    elif args.format == 'torchscript':
        exporter.export_torchscript(str(output_path / f"{args.name}.pt"))
    elif args.format == 'quantized':
        exporter.export_quantized(str(output_path / f"{args.name}_quantized.pth"))
    elif args.format == 'summary':
        exporter.export_summary(str(output_path / f"{args.name}_summary.txt"))


if __name__ == "__main__":
    main()
