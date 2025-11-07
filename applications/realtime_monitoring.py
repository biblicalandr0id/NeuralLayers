"""
Real-Time Monitoring and Anomaly Detection

Production-ready monitoring system using the UMI layer for:
- System health monitoring
- Anomaly detection
- Alert generation
- Dashboard visualization
- Historical trend analysis

Can monitor:
- Server metrics (CPU, memory, disk, network)
- Application performance
- Custom metrics

Usage:
    python realtime_monitoring.py --source simulated
    python realtime_monitoring.py --source file --input metrics.csv
    python realtime_monitoring.py --source api --endpoint http://metrics-api.com
"""

import argparse
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

from umi_layer import UMI_Network, UMI_Layer
from utils import Logger, Config

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Visualization disabled.")


class MetricsBuffer:
    """Ring buffer for metrics history."""

    def __init__(self, max_size: int = 1000):
        """
        Initialize buffer.

        Args:
            max_size: Maximum number of samples to keep
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)

    def add(self, metrics: np.ndarray, timestamp: Optional[datetime] = None):
        """Add metrics to buffer."""
        self.buffer.append(metrics)
        self.timestamps.append(timestamp or datetime.now())

    def get_recent(self, n: int) -> Tuple[np.ndarray, List[datetime]]:
        """Get n most recent samples."""
        n = min(n, len(self.buffer))
        metrics = np.array(list(self.buffer)[-n:])
        timestamps = list(self.timestamps)[-n:]
        return metrics, timestamps

    def get_all(self) -> Tuple[np.ndarray, List[datetime]]:
        """Get all samples."""
        return np.array(list(self.buffer)), list(self.timestamps)


class RealTimeMonitor:
    """Real-time monitoring and anomaly detection system."""

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize monitor.

        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.logger = Logger("RealTimeMonitor", self.config)

        # UMI model
        self.umi_network = UMI_Network(
            critical_threshold=self.config.get('umi.critical_threshold', 1.0),
            warning_threshold=self.config.get('umi.warning_threshold', 0.7)
        )

        # Metrics buffer
        self.buffer = MetricsBuffer(max_size=self.config.get('monitoring.buffer_size', 1000))

        # Statistics
        self.baseline = None
        self.stats = {
            'total_samples': 0,
            'normal_count': 0,
            'warning_count': 0,
            'critical_count': 0
        }

        # Alert history
        self.alerts = []

    def compute_metrics(self, raw_data: Dict[str, float]) -> np.ndarray:
        """
        Compute UMI input metrics from raw data.

        Args:
            raw_data: Dictionary of raw metric values

        Returns:
            Array of [DeltaR, T, V, A] for UMI
        """
        # Compute baseline if not exists
        if self.baseline is None:
            self.baseline = {k: v for k, v in raw_data.items()}
            return np.array([0.0, 0.0, 0.0, 0.0])

        # Delta R: Relative deviation from baseline
        deviations = [(raw_data[k] - self.baseline[k]) / (self.baseline[k] + 1e-8)
                     for k in raw_data.keys()]
        delta_r = np.mean(np.abs(deviations))

        # T: Trend coefficient (requires history)
        recent_data, _ = self.buffer.get_recent(10)
        if len(recent_data) >= 2:
            trend = np.mean(np.diff(recent_data[:, 0]))  # Trend in first metric
        else:
            trend = 0.0

        # V: Coefficient of variation
        if len(recent_data) > 0:
            variation = np.std(recent_data[:, 0]) / (np.mean(recent_data[:, 0]) + 1e-8)
        else:
            variation = 0.0

        # A: Anomaly score (distance from baseline)
        anomaly = np.max(np.abs(deviations))

        return np.array([delta_r, trend, variation, anomaly])

    def process_sample(self, raw_data: Dict[str, float]) -> Dict:
        """
        Process a single sample.

        Args:
            raw_data: Dictionary of raw metric values

        Returns:
            Dictionary with UMI score, alert level, and metrics
        """
        # Compute UMI metrics
        umi_metrics = self.compute_metrics(raw_data)

        # Add to buffer
        self.buffer.add(umi_metrics)

        # Compute UMI score
        umi_input = torch.tensor([umi_metrics], dtype=torch.float32)
        umi_score, alert_level = self.umi_network(umi_input, return_alerts=True)

        umi_score_value = umi_score.item()
        alert_value = alert_level.item()

        # Update statistics
        self.stats['total_samples'] += 1
        if alert_value == 0:
            self.stats['normal_count'] += 1
        elif alert_value == 1:
            self.stats['warning_count'] += 1
        else:
            self.stats['critical_count'] += 1

        # Generate alert if needed
        if alert_value > 0:
            alert = {
                'timestamp': datetime.now(),
                'level': alert_value,
                'umi_score': umi_score_value,
                'metrics': umi_metrics,
                'raw_data': raw_data
            }
            self.alerts.append(alert)

        return {
            'umi_score': umi_score_value,
            'alert_level': alert_value,
            'umi_metrics': umi_metrics,
            'raw_data': raw_data
        }

    def monitor_stream(self, data_source, duration: Optional[float] = None,
                      interval: float = 1.0):
        """
        Monitor a data stream.

        Args:
            data_source: Callable that returns metric dictionary
            duration: Monitoring duration in seconds (None = infinite)
            interval: Sampling interval in seconds
        """
        self.logger.info("Starting real-time monitoring...")
        self.logger.info(f"Sampling interval: {interval}s")

        start_time = time.time()
        sample_count = 0

        try:
            while True:
                # Check duration
                if duration and (time.time() - start_time) > duration:
                    break

                # Get sample
                raw_data = data_source()

                # Process
                result = self.process_sample(raw_data)

                # Log
                alert_names = ['NORMAL', 'WARNING', 'CRITICAL']
                alert_name = alert_names[result['alert_level']]

                if result['alert_level'] > 0:
                    self.logger.warning(
                        f"[{sample_count:04d}] UMI: {result['umi_score']:7.4f} | "
                        f"Alert: {alert_name}"
                    )
                else:
                    if sample_count % 10 == 0:  # Log every 10th normal sample
                        self.logger.info(
                            f"[{sample_count:04d}] UMI: {result['umi_score']:7.4f} | "
                            f"Status: {alert_name}"
                        )

                sample_count += 1
                time.sleep(interval)

        except KeyboardInterrupt:
            self.logger.info("\nMonitoring stopped by user")

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print monitoring summary."""
        print("\n" + "=" * 70)
        print(" " * 20 + "MONITORING SUMMARY")
        print("=" * 70)

        total = self.stats['total_samples']
        normal_pct = (self.stats['normal_count'] / total * 100) if total > 0 else 0
        warning_pct = (self.stats['warning_count'] / total * 100) if total > 0 else 0
        critical_pct = (self.stats['critical_count'] / total * 100) if total > 0 else 0

        print(f"Total samples: {total}")
        print(f"\nAlert Distribution:")
        print(f"  🟢 Normal: {self.stats['normal_count']:6d} ({normal_pct:5.1f}%)")
        print(f"  🟡 Warning: {self.stats['warning_count']:6d} ({warning_pct:5.1f}%)")
        print(f"  🔴 Critical: {self.stats['critical_count']:6d} ({critical_pct:5.1f}%)")

        print(f"\nTotal alerts: {len(self.alerts)}")

        if self.alerts:
            print(f"\nRecent alerts (last 5):")
            for alert in self.alerts[-5:]:
                alert_names = ['NORMAL', 'WARNING', 'CRITICAL']
                print(f"  [{alert['timestamp'].strftime('%H:%M:%S')}] "
                      f"{alert_names[alert['level']]}: UMI = {alert['umi_score']:.4f}")

        print("=" * 70)

    def visualize_live(self, data_source, update_interval: int = 1000):
        """
        Live visualization dashboard.

        Args:
            data_source: Callable that returns metric dictionary
            update_interval: Update interval in milliseconds
        """
        if not MATPLOTLIB_AVAILABLE:
            self.logger.error("matplotlib not available for visualization")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Real-Time Monitoring Dashboard', fontsize=16, fontweight='bold')

        # Data containers
        umi_scores = deque(maxlen=100)
        timestamps = deque(maxlen=100)
        alerts_timeline = deque(maxlen=100)

        def update(frame):
            # Get new sample
            raw_data = data_source()
            result = self.process_sample(raw_data)

            # Update data
            umi_scores.append(result['umi_score'])
            timestamps.append(len(umi_scores))
            alerts_timeline.append(result['alert_level'])

            # Clear axes
            for ax in axes.flat:
                ax.clear()

            # Plot 1: UMI Score over time
            axes[0, 0].plot(list(timestamps), list(umi_scores), 'b-', linewidth=2)
            axes[0, 0].axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Warning')
            axes[0, 0].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Critical')
            axes[0, 0].set_title('UMI Score Timeline')
            axes[0, 0].set_xlabel('Sample')
            axes[0, 0].set_ylabel('UMI Score')
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)

            # Plot 2: Alert levels
            axes[0, 1].scatter(list(timestamps), list(alerts_timeline),
                             c=list(alerts_timeline), cmap='RdYlGn_r',
                             s=50, alpha=0.6, vmin=0, vmax=2)
            axes[0, 1].set_title('Alert Levels')
            axes[0, 1].set_xlabel('Sample')
            axes[0, 1].set_ylabel('Alert Level')
            axes[0, 1].set_yticks([0, 1, 2])
            axes[0, 1].set_yticklabels(['Normal', 'Warning', 'Critical'])
            axes[0, 1].grid(alpha=0.3)

            # Plot 3: Statistics
            total = self.stats['total_samples']
            if total > 0:
                values = [
                    self.stats['normal_count'],
                    self.stats['warning_count'],
                    self.stats['critical_count']
                ]
                labels = ['Normal', 'Warning', 'Critical']
                colors = ['green', 'orange', 'red']

                axes[1, 0].bar(labels, values, color=colors, alpha=0.7)
                axes[1, 0].set_title('Alert Distribution')
                axes[1, 0].set_ylabel('Count')
                axes[1, 0].grid(axis='y', alpha=0.3)

            # Plot 4: Current status
            axes[1, 1].text(0.5, 0.6, f"UMI Score:\n{umi_scores[-1]:.4f}",
                          ha='center', va='center', fontsize=20, fontweight='bold')

            alert_names = ['NORMAL', 'WARNING', 'CRITICAL']
            alert_colors = ['green', 'orange', 'red']
            current_alert = alerts_timeline[-1] if alerts_timeline else 0

            axes[1, 1].text(0.5, 0.3, alert_names[current_alert],
                          ha='center', va='center', fontsize=24,
                          fontweight='bold', color=alert_colors[current_alert])

            axes[1, 1].set_title('Current Status')
            axes[1, 1].axis('off')

            plt.tight_layout()

        # Animation
        ani = animation.FuncAnimation(fig, update, interval=update_interval,
                                     cache_frame_data=False)
        plt.show()


def simulated_data_source() -> Dict[str, float]:
    """Generate simulated metric data."""
    # Simulate metrics with occasional anomalies
    baseline = {
        'cpu': 45.0,
        'memory': 60.0,
        'disk_io': 30.0,
        'network': 20.0
    }

    # Add noise
    noise = np.random.randn(4) * 5.0

    # Occasional spike
    if np.random.rand() < 0.05:  # 5% chance
        noise += np.random.randn(4) * 30.0

    return {
        'cpu': baseline['cpu'] + noise[0],
        'memory': baseline['memory'] + noise[1],
        'disk_io': baseline['disk_io'] + noise[2],
        'network': baseline['network'] + noise[3]
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Real-Time Monitoring System')
    parser.add_argument('--source', type=str, default='simulated',
                       choices=['simulated', 'file', 'api'],
                       help='Data source type')
    parser.add_argument('--duration', type=float, default=60,
                       help='Monitoring duration in seconds')
    parser.add_argument('--interval', type=float, default=1.0,
                       help='Sampling interval in seconds')
    parser.add_argument('--visualize', action='store_true',
                       help='Enable live visualization')

    args = parser.parse_args()

    # Create monitor
    monitor = RealTimeMonitor()

    # Select data source
    if args.source == 'simulated':
        data_source = simulated_data_source
    else:
        raise NotImplementedError(f"Data source '{args.source}' not implemented")

    # Start monitoring
    if args.visualize:
        monitor.visualize_live(data_source, update_interval=int(args.interval * 1000))
    else:
        monitor.monitor_stream(data_source, duration=args.duration, interval=args.interval)


if __name__ == "__main__":
    main()
