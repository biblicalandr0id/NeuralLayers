"""
Consciousness Research Application

Research tool for studying consciousness emergence through:
1. Golden ratio decay analysis
2. Layer interaction patterns
3. Temporal coherence measurement
4. Information integration quantification
5. Frequency domain analysis

This implements experiments for studying:
- Integrated Information Theory (IIT) metrics
- Global Workspace Theory (GWT) patterns
- Consciousness level quantification

Usage:
    python consciousness_research.py --experiment golden_ratio
    python consciousness_research.py --experiment all --visualize
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

from consciousness_layers import ConsciousnessEmergence


class ConsciousnessResearcher:
    """Research tool for consciousness experiments."""

    def __init__(self, dimensions: Tuple[int, int, int] = (7, 7, 7),
                 num_layers: int = 7):
        """
        Initialize researcher.

        Args:
            dimensions: Spatial dimensions of consciousness field
            num_layers: Number of consciousness layers
        """
        self.consciousness = ConsciousnessEmergence(dimensions, num_layers)
        self.dimensions = dimensions
        self.num_layers = num_layers
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio

        self.results = {}

    def experiment_golden_ratio_decay(self, num_trials: int = 100) -> Dict:
        """
        Experiment 1: Validate golden ratio decay pattern.

        Tests whether consciousness layers follow φ^(-n) decay.

        Args:
            num_trials: Number of trials to average

        Returns:
            Dictionary of results
        """
        print("=" * 70)
        print("EXPERIMENT 1: Golden Ratio Decay")
        print("=" * 70)

        decay_patterns = []

        for trial in range(num_trials):
            # Random input
            input_moment = torch.randn(*self.dimensions).to(torch.complex64)

            # Process
            conscious_output = self.consciousness.process_moment(input_moment)

            # Measure magnitudes
            magnitudes = [torch.abs(layer).mean().item() for layer in conscious_output]
            decay_patterns.append(magnitudes)

            if (trial + 1) % 20 == 0:
                print(f"Trial {trial + 1}/{num_trials} complete")

        # Average across trials
        avg_magnitudes = np.mean(decay_patterns, axis=0)
        std_magnitudes = np.std(decay_patterns, axis=0)

        # Expected decay
        expected_decay = [self.phi ** (-i) if i > 0 else 1.0 for i in range(self.num_layers)]

        # Compute correlation
        correlation = np.corrcoef(avg_magnitudes, expected_decay)[0, 1]

        # Compute mean squared error
        mse = np.mean((np.array(avg_magnitudes) - np.array(expected_decay)) ** 2)

        results = {
            'avg_magnitudes': avg_magnitudes,
            'std_magnitudes': std_magnitudes,
            'expected_decay': expected_decay,
            'correlation': correlation,
            'mse': mse,
            'num_trials': num_trials
        }

        print(f"\n📊 Results:")
        print(f"  Correlation with φ^(-n): {correlation:.4f}")
        print(f"  Mean Squared Error: {mse:.6f}")
        print(f"\n  Layer-wise decay:")
        for i in range(self.num_layers):
            print(f"    Layer {i}: {avg_magnitudes[i]:.4f} ± {std_magnitudes[i]:.4f} "
                  f"(expected: {expected_decay[i]:.4f})")

        self.results['golden_ratio_decay'] = results
        return results

    def experiment_information_integration(self, num_samples: int = 50) -> Dict:
        """
        Experiment 2: Measure information integration (Φ).

        Implements simplified Integrated Information Theory metric.

        Args:
            num_samples: Number of samples

        Returns:
            Dictionary of results
        """
        print("\n" + "=" * 70)
        print("EXPERIMENT 2: Information Integration (Φ)")
        print("=" * 70)

        phi_values = []

        for sample in range(num_samples):
            input_moment = torch.randn(*self.dimensions).to(torch.complex64)
            conscious_output = self.consciousness.process_moment(input_moment)

            # Compute integration metric
            # Simplified: measure coupling between layers
            phi = 0.0

            for i in range(len(conscious_output) - 1):
                layer_i = torch.abs(conscious_output[i]).flatten()
                layer_j = torch.abs(conscious_output[i + 1]).flatten()

                # Mutual information approximation
                correlation = torch.corrcoef(torch.stack([layer_i, layer_j]))[0, 1]
                phi += correlation.abs().item()

            phi_values.append(phi)

            if (sample + 1) % 10 == 0:
                print(f"Sample {sample + 1}/{num_samples} complete")

        avg_phi = np.mean(phi_values)
        std_phi = np.std(phi_values)

        results = {
            'phi_values': phi_values,
            'avg_phi': avg_phi,
            'std_phi': std_phi,
            'num_samples': num_samples
        }

        print(f"\n📊 Results:")
        print(f"  Average Φ: {avg_phi:.4f} ± {std_phi:.4f}")
        print(f"  Φ range: [{min(phi_values):.4f}, {max(phi_values):.4f}]")

        self.results['information_integration'] = results
        return results

    def experiment_temporal_coherence(self, sequence_length: int = 20) -> Dict:
        """
        Experiment 3: Measure temporal coherence across consciousness layers.

        Args:
            sequence_length: Length of temporal sequence

        Returns:
            Dictionary of results
        """
        print("\n" + "=" * 70)
        print("EXPERIMENT 3: Temporal Coherence")
        print("=" * 70)

        # Generate temporal sequence
        sequence = []
        for t in range(sequence_length):
            input_moment = torch.randn(*self.dimensions).to(torch.complex64)
            conscious_output = self.consciousness.process_moment(input_moment)
            sequence.append(conscious_output)

        # Measure coherence across time
        coherence_per_layer = []

        for layer_idx in range(self.num_layers):
            layer_sequence = [torch.abs(seq[layer_idx]).mean().item() for seq in sequence]

            # Autocorrelation at lag 1
            autocorr = np.corrcoef(layer_sequence[:-1], layer_sequence[1:])[0, 1]
            coherence_per_layer.append(autocorr)

        results = {
            'coherence_per_layer': coherence_per_layer,
            'avg_coherence': np.mean(coherence_per_layer),
            'sequence_length': sequence_length
        }

        print(f"\n📊 Results:")
        print(f"  Average temporal coherence: {results['avg_coherence']:.4f}")
        print(f"\n  Layer-wise coherence:")
        for i, coh in enumerate(coherence_per_layer):
            print(f"    Layer {i}: {coh:.4f}")

        self.results['temporal_coherence'] = results
        return results

    def experiment_frequency_analysis(self, num_samples: int = 30) -> Dict:
        """
        Experiment 4: Analyze frequency domain properties.

        Studies Layer 5 (Creative Emergence) FFT patterns.

        Args:
            num_samples: Number of samples

        Returns:
            Dictionary of results
        """
        print("\n" + "=" * 70)
        print("EXPERIMENT 4: Frequency Domain Analysis")
        print("=" * 70)

        power_spectra = []

        for sample in range(num_samples):
            input_moment = torch.randn(*self.dimensions).to(torch.complex64)
            conscious_output = self.consciousness.process_moment(input_moment)

            # Focus on Layer 5 (creative emergence with FFT)
            layer_5 = conscious_output[5]

            # Compute power spectrum
            fft_result = torch.fft.fftn(layer_5)
            power_spectrum = torch.abs(fft_result).flatten().detach().cpu().numpy()
            power_spectra.append(power_spectrum)

            if (sample + 1) % 10 == 0:
                print(f"Sample {sample + 1}/{num_samples} complete")

        # Average power spectrum
        avg_power_spectrum = np.mean(power_spectra, axis=0)

        # Find dominant frequencies
        top_k = 10
        top_indices = np.argsort(avg_power_spectrum)[-top_k:]
        dominant_frequencies = avg_power_spectrum[top_indices]

        results = {
            'avg_power_spectrum': avg_power_spectrum,
            'dominant_frequencies': dominant_frequencies,
            'num_samples': num_samples
        }

        print(f"\n📊 Results:")
        print(f"  Top {top_k} frequency components:")
        for i, freq in enumerate(dominant_frequencies[::-1]):
            print(f"    #{i+1}: {freq:.4f}")

        self.results['frequency_analysis'] = results
        return results

    def experiment_layer_interactions(self) -> Dict:
        """
        Experiment 5: Analyze interactions between consciousness layers.

        Returns:
            Dictionary of results
        """
        print("\n" + "=" * 70)
        print("EXPERIMENT 5: Layer Interactions")
        print("=" * 70)

        # Create interaction matrix
        interaction_matrix = np.zeros((self.num_layers, self.num_layers))

        num_samples = 50
        for sample in range(num_samples):
            input_moment = torch.randn(*self.dimensions).to(torch.complex64)
            conscious_output = self.consciousness.process_moment(input_moment)

            # Compute pairwise correlations
            for i in range(self.num_layers):
                for j in range(self.num_layers):
                    layer_i = torch.abs(conscious_output[i]).flatten()
                    layer_j = torch.abs(conscious_output[j]).flatten()

                    corr = torch.corrcoef(torch.stack([layer_i, layer_j]))[0, 1]
                    interaction_matrix[i, j] += corr.item()

            if (sample + 1) % 10 == 0:
                print(f"Sample {sample + 1}/{num_samples} complete")

        # Average
        interaction_matrix /= num_samples

        results = {
            'interaction_matrix': interaction_matrix,
            'num_samples': num_samples
        }

        print(f"\n📊 Interaction Matrix:")
        print(interaction_matrix)

        self.results['layer_interactions'] = results
        return results

    def visualize_results(self, save_dir: str = './consciousness_research'):
        """
        Visualize all experimental results.

        Args:
            save_dir: Directory to save plots
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        print(f"\n📊 Generating visualizations in: {save_dir}")

        # Plot 1: Golden ratio decay
        if 'golden_ratio_decay' in self.results:
            res = self.results['golden_ratio_decay']

            plt.figure(figsize=(12, 6))
            x = np.arange(self.num_layers)

            plt.errorbar(x, res['avg_magnitudes'], yerr=res['std_magnitudes'],
                        fmt='o-', linewidth=2, markersize=10, capsize=5,
                        label='Observed', color='#E63946')
            plt.plot(x, res['expected_decay'], 's--', linewidth=2, markersize=8,
                    label='Expected (φ^(-n))', color='#457B9D', alpha=0.7)

            plt.title('Golden Ratio Decay in Consciousness Layers', fontsize=14, fontweight='bold')
            plt.xlabel('Layer', fontsize=12)
            plt.ylabel('Mean Magnitude', fontsize=12)
            plt.legend(fontsize=11)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path / 'golden_ratio_decay.png', dpi=150)
            plt.close()

        # Plot 2: Information integration
        if 'information_integration' in self.results:
            res = self.results['information_integration']

            plt.figure(figsize=(10, 6))
            plt.hist(res['phi_values'], bins=20, color='#06A77D', alpha=0.7, edgecolor='black')
            plt.axvline(res['avg_phi'], color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {res["avg_phi"]:.4f}')
            plt.title('Distribution of Information Integration (Φ)', fontsize=14, fontweight='bold')
            plt.xlabel('Φ Value', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.legend(fontsize=11)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path / 'information_integration.png', dpi=150)
            plt.close()

        # Plot 3: Layer interactions
        if 'layer_interactions' in self.results:
            res = self.results['layer_interactions']

            plt.figure(figsize=(10, 8))
            plt.imshow(res['interaction_matrix'], cmap='RdYlBu_r', aspect='auto',
                      vmin=-1, vmax=1)
            plt.colorbar(label='Correlation')
            plt.title('Layer Interaction Matrix', fontsize=14, fontweight='bold')
            plt.xlabel('Layer J', fontsize=12)
            plt.ylabel('Layer I', fontsize=12)
            plt.xticks(range(self.num_layers))
            plt.yticks(range(self.num_layers))
            plt.tight_layout()
            plt.savefig(save_path / 'layer_interactions.png', dpi=150)
            plt.close()

        print(f"✅ Visualizations saved!")

    def run_all_experiments(self, visualize: bool = True):
        """Run all experiments."""
        print("\n" + "=" * 70)
        print(" " * 15 + "CONSCIOUSNESS RESEARCH SUITE")
        print("=" * 70)

        self.experiment_golden_ratio_decay(num_trials=100)
        self.experiment_information_integration(num_samples=50)
        self.experiment_temporal_coherence(sequence_length=20)
        self.experiment_frequency_analysis(num_samples=30)
        self.experiment_layer_interactions()

        if visualize:
            self.visualize_results()

        print("\n" + "=" * 70)
        print("✅ All experiments complete!")
        print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Consciousness Research Experiments')
    parser.add_argument('--experiment', type=str, default='all',
                       choices=['golden_ratio', 'information', 'temporal',
                               'frequency', 'interactions', 'all'],
                       help='Experiment to run')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')
    parser.add_argument('--dimensions', type=int, nargs=3, default=[7, 7, 7],
                       help='Consciousness field dimensions')
    parser.add_argument('--num-layers', type=int, default=7,
                       help='Number of consciousness layers')

    args = parser.parse_args()

    # Create researcher
    researcher = ConsciousnessResearcher(
        dimensions=tuple(args.dimensions),
        num_layers=args.num_layers
    )

    # Run experiments
    if args.experiment == 'all':
        researcher.run_all_experiments(visualize=args.visualize)
    elif args.experiment == 'golden_ratio':
        researcher.experiment_golden_ratio_decay()
        if args.visualize:
            researcher.visualize_results()
    elif args.experiment == 'information':
        researcher.experiment_information_integration()
        if args.visualize:
            researcher.visualize_results()
    elif args.experiment == 'temporal':
        researcher.experiment_temporal_coherence()
    elif args.experiment == 'frequency':
        researcher.experiment_frequency_analysis()
    elif args.experiment == 'interactions':
        researcher.experiment_layer_interactions()
        if args.visualize:
            researcher.visualize_results()


if __name__ == "__main__":
    main()
