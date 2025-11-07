"""
Basic Training Example

Demonstrates how to train a NeuralLayers model on a simple task.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path

from logicalbrain_network import UnifiedBrainNetwork


def generate_synthetic_data(num_samples=1000, input_dim=128, output_dim=64):
    """Generate synthetic data for training"""
    X = torch.randn(num_samples, input_dim)
    # Simple target: sum of first half should be positive, second half negative
    y = torch.zeros(num_samples, output_dim)
    y[:, :output_dim//2] = torch.abs(X[:, :input_dim//2].mean(dim=1, keepdim=True))
    y[:, output_dim//2:] = -torch.abs(X[:, input_dim//2:].mean(dim=1, keepdim=True))

    return X, y


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_X, batch_y in dataloader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(batch_X)

        # Compute loss
        loss = criterion(output['output'][:, :batch_y.size(1)], batch_y)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            output = model(batch_X)
            loss = criterion(output['output'][:, :batch_y.size(1)], batch_y)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def main():
    """Run training example"""

    print("=" * 70)
    print("NeuralLayers - Basic Training Example")
    print("=" * 70)

    # Configuration
    input_dim = 128
    hidden_dim = 256
    output_dim = 64
    num_samples = 1000
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\nConfiguration:")
    print(f"  Device:           {device}")
    print(f"  Input dimension:  {input_dim}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Output dimension: {output_dim}")
    print(f"  Batch size:       {batch_size}")
    print(f"  Learning rate:    {learning_rate}")
    print(f"  Epochs:           {num_epochs}")

    # Generate data
    print(f"\n{'Generating synthetic data':<40}", end="")
    X_train, y_train = generate_synthetic_data(num_samples, input_dim, output_dim)
    X_val, y_val = generate_synthetic_data(num_samples // 5, input_dim, output_dim)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"✅ ({num_samples} train, {num_samples//5} val)")

    # Initialize model
    print(f"{'Initializing model':<40}", end="")
    model = UnifiedBrainNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=4
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ ({total_params:,} parameters)")

    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print("\n" + "=" * 70)
    print("Training Progress")
    print("=" * 70)

    train_losses = []
    val_losses = []

    print(f"\n{'Epoch':<10} {'Train Loss':<15} {'Val Loss':<15} {'Status':<20}")
    print("-" * 70)

    best_val_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        # Validate
        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # Check improvement
        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            status = "✨ Best model!"

        print(f"{epoch:<10} {train_loss:<15.6f} {val_loss:<15.6f} {status:<20}")

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"  Best validation loss: {best_val_loss:.6f}")
    print(f"  Final train loss:     {train_losses[-1]:.6f}")
    print(f"  Final val loss:       {val_losses[-1]:.6f}")

    # Plot results
    print(f"\n{'Generating training plots':<40}", end="")

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, 'o-', label='Train Loss', linewidth=2)
    plt.plot(range(1, num_epochs + 1), val_losses, 's-', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Progress', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_losses, 'o-', linewidth=2)
    plt.plot(range(1, num_epochs + 1), val_losses, 's-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (log scale)', fontsize=12)
    plt.title('Training Progress (Log Scale)', fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    Path('outputs').mkdir(exist_ok=True)
    plt.savefig('outputs/training_progress.png', dpi=150, bbox_inches='tight')
    print("✅ Saved to outputs/training_progress.png")

    # Save model
    print(f"{'Saving trained model':<40}", end="")
    Path('checkpoints').mkdir(exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'config': {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim,
        }
    }, 'checkpoints/basic_training.pth')
    print("✅ Saved to checkpoints/basic_training.pth")

    print("\n✨ Training example completed successfully!")
    print("\nNext steps:")
    print("  1. View training plot: outputs/training_progress.png")
    print("  2. Load model: torch.load('checkpoints/basic_training.pth')")
    print("  3. Try different hyperparameters")
    print("  4. Explore advanced training: python train.py")


if __name__ == '__main__':
    main()
