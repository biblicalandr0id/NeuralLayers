"""
Batch Inference Engine for NeuralLayers

Features:
- Efficient batch processing
- Multi-GPU support
- Progress tracking
- Result caching
- Fault tolerance
- Resumable processing
"""

import os
import json
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, List, Iterator
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time


@dataclass
class BatchInferenceConfig:
    """Configuration for batch inference"""

    # Data
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True

    # Processing
    device: str = "cpu"
    use_amp: bool = False  # Automatic mixed precision
    max_batch_size: int = 128  # For dynamic batching

    # Output
    output_dir: str = "inference_results"
    save_format: str = "npy"  # npy, npz, pickle, json
    save_interval: int = 1000  # Save every N batches

    # Fault tolerance
    checkpoint_interval: int = 100  # Checkpoint every N batches
    resume_from_checkpoint: bool = True

    # Multi-GPU
    distributed: bool = False
    world_size: int = 1
    rank: int = 0


class InferenceDataset(Dataset):
    """
    Dataset for batch inference

    Supports:
    - NumPy arrays
    - HDF5 files
    - List of file paths
    """

    def __init__(
        self,
        data: Any,
        preprocess_fn: Optional[callable] = None
    ):
        self.data = data
        self.preprocess_fn = preprocess_fn

        # Determine data type
        if isinstance(data, np.ndarray):
            self.data_type = "array"
            self.length = len(data)

        elif isinstance(data, list):
            self.data_type = "list"
            self.length = len(data)

        elif isinstance(data, str) and data.endswith('.h5'):
            import h5py
            self.data_type = "hdf5"
            self.h5_file = h5py.File(data, 'r')
            self.h5_dataset = self.h5_file['data']
            self.length = len(self.h5_dataset)

        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        # Load data
        if self.data_type == "array":
            item = self.data[idx]

        elif self.data_type == "list":
            item = self.data[idx]

        elif self.data_type == "hdf5":
            item = self.h5_dataset[idx]

        # Convert to tensor
        if isinstance(item, np.ndarray):
            item = torch.from_numpy(item).float()
        elif not isinstance(item, torch.Tensor):
            item = torch.tensor(item, dtype=torch.float32)

        # Preprocess
        if self.preprocess_fn:
            item = self.preprocess_fn(item)

        return idx, item

    def __del__(self):
        if hasattr(self, 'h5_file'):
            self.h5_file.close()


class BatchInferenceEngine:
    """
    Batch inference engine with fault tolerance and progress tracking
    """

    def __init__(
        self,
        model: nn.Module,
        config: BatchInferenceConfig
    ):
        self.model = model
        self.config = config

        # Setup device
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.model.eval()

        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint tracking
        self.checkpoint_path = self.output_dir / "checkpoint.json"
        self.processed_indices = set()

        # Results buffer
        self.results_buffer: Dict[int, np.ndarray] = {}

    def load_checkpoint(self) -> Dict[str, Any]:
        """Load checkpoint if exists"""

        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, 'r') as f:
                checkpoint = json.load(f)

            self.processed_indices = set(checkpoint['processed_indices'])

            print(f"✅ Resumed from checkpoint: {len(self.processed_indices)} samples processed")

            return checkpoint

        return {'processed_indices': []}

    def save_checkpoint(self, batch_idx: int, total_batches: int):
        """Save checkpoint"""

        checkpoint = {
            'batch_idx': batch_idx,
            'total_batches': total_batches,
            'processed_indices': list(self.processed_indices),
            'timestamp': time.time()
        }

        with open(self.checkpoint_path, 'w') as f:
            json.dump(checkpoint, f)

    def save_results(
        self,
        results: Dict[int, np.ndarray],
        suffix: str = ""
    ):
        """Save results to disk"""

        if not results:
            return

        output_file = self.output_dir / f"results{suffix}.{self.config.save_format}"

        if self.config.save_format == "npy":
            # Save as single numpy array (must be sorted)
            sorted_items = sorted(results.items())
            indices = np.array([idx for idx, _ in sorted_items])
            outputs = np.array([out for _, out in sorted_items])

            np.save(output_file, outputs)
            np.save(self.output_dir / f"indices{suffix}.npy", indices)

        elif self.config.save_format == "npz":
            # Save as compressed numpy
            np.savez_compressed(output_file, **{str(k): v for k, v in results.items()})

        elif self.config.save_format == "pickle":
            # Save as pickle
            with open(output_file, 'wb') as f:
                pickle.dump(results, f)

        elif self.config.save_format == "json":
            # Save as JSON (convert to lists)
            json_results = {str(k): v.tolist() for k, v in results.items()}
            with open(output_file, 'w') as f:
                json.dump(json_results, f)

        print(f"💾 Saved {len(results)} results to {output_file}")

    @torch.no_grad()
    def infer_batch(
        self,
        inputs: torch.Tensor
    ) -> torch.Tensor:
        """Run inference on a batch"""

        inputs = inputs.to(self.device)

        # Mixed precision
        if self.config.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
        else:
            outputs = self.model(inputs)

        # Handle dict output
        if isinstance(outputs, dict):
            outputs = outputs['output']

        return outputs.cpu()

    def run(
        self,
        dataset: Dataset,
        resume: bool = True
    ) -> Dict[int, np.ndarray]:
        """
        Run batch inference

        Returns:
            Dictionary mapping sample indices to outputs
        """

        # Load checkpoint if resuming
        if resume and self.config.resume_from_checkpoint:
            self.load_checkpoint()

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            shuffle=False
        )

        total_batches = len(dataloader)
        all_results: Dict[int, np.ndarray] = {}

        # Progress bar
        pbar = tqdm(
            enumerate(dataloader),
            total=total_batches,
            desc="Batch Inference"
        )

        for batch_idx, (indices, inputs) in pbar:
            # Skip if already processed
            if resume and all(idx.item() in self.processed_indices for idx in indices):
                continue

            # Run inference
            outputs = self.infer_batch(inputs)

            # Store results
            for idx, output in zip(indices, outputs):
                idx_val = idx.item()
                all_results[idx_val] = output.numpy()
                self.processed_indices.add(idx_val)
                self.results_buffer[idx_val] = output.numpy()

            # Save intermediate results
            if (batch_idx + 1) % self.config.save_interval == 0:
                self.save_results(
                    self.results_buffer,
                    suffix=f"_batch_{batch_idx+1}"
                )
                self.results_buffer = {}

            # Save checkpoint
            if (batch_idx + 1) % self.config.checkpoint_interval == 0:
                self.save_checkpoint(batch_idx, total_batches)

            # Update progress
            pbar.set_postfix({
                'processed': len(self.processed_indices),
                'total': len(dataset)
            })

        # Save remaining results
        if self.results_buffer:
            self.save_results(self.results_buffer, suffix="_final")

        # Save all results
        self.save_results(all_results, suffix="")

        # Clean up checkpoint
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()

        print(f"✅ Inference complete: {len(all_results)} samples processed")

        return all_results

    def run_streaming(
        self,
        dataset: Dataset,
        callback: Optional[callable] = None
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Run batch inference in streaming mode (yields results as they're computed)

        Args:
            dataset: Input dataset
            callback: Optional callback function called with (index, output) for each sample

        Yields:
            (index, output) tuples
        """

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            shuffle=False
        )

        for indices, inputs in dataloader:
            outputs = self.infer_batch(inputs)

            for idx, output in zip(indices, outputs):
                idx_val = idx.item()
                output_np = output.numpy()

                if callback:
                    callback(idx_val, output_np)

                yield idx_val, output_np


# ============================================================================
# Distributed Batch Inference
# ============================================================================

class DistributedBatchInferenceEngine(BatchInferenceEngine):
    """Batch inference engine with multi-GPU support"""

    def __init__(
        self,
        model: nn.Module,
        config: BatchInferenceConfig
    ):
        super().__init__(model, config)

        if config.distributed:
            # Wrap model in DistributedDataParallel
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[config.rank]
            )

    def run(
        self,
        dataset: Dataset,
        resume: bool = True
    ) -> Dict[int, np.ndarray]:
        """Run distributed batch inference"""

        from torch.utils.data.distributed import DistributedSampler

        # Create distributed sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.config.world_size,
            rank=self.config.rank,
            shuffle=False
        )

        # Create dataloader with sampler
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )

        all_results: Dict[int, np.ndarray] = {}

        pbar = tqdm(
            dataloader,
            desc=f"Batch Inference (Rank {self.config.rank})",
            disable=self.config.rank != 0
        )

        for indices, inputs in pbar:
            outputs = self.infer_batch(inputs)

            for idx, output in zip(indices, outputs):
                all_results[idx.item()] = output.numpy()

        # Save rank-specific results
        self.save_results(all_results, suffix=f"_rank_{self.config.rank}")

        return all_results


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from logicalbrain_network import UnifiedBrainLogicNetwork

    # Create model
    model = UnifiedBrainLogicNetwork(
        input_dim=128,
        hidden_dim=128,
        output_dim=64
    )

    # Create synthetic dataset
    data = np.random.randn(1000, 128).astype(np.float32)
    dataset = InferenceDataset(data)

    # Configure inference
    config = BatchInferenceConfig(
        batch_size=32,
        num_workers=4,
        output_dir="inference_results",
        save_format="npz",
        checkpoint_interval=10
    )

    # Run batch inference
    engine = BatchInferenceEngine(model, config)

    print("="*80)
    print("BATCH INFERENCE DEMO")
    print("="*80)
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {config.batch_size}")
    print(f"Output directory: {config.output_dir}")
    print("="*80)

    results = engine.run(dataset, resume=True)

    print(f"\n✅ Processed {len(results)} samples")
    print(f"📊 Output shape: {list(results.values())[0].shape}")
