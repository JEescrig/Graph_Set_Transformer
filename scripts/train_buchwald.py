"""
Training script for Buchwald-Hartwig reaction yield prediction.
Regression task using set-based models on reactant molecules.
"""
import torch
import torch.nn.functional as F
import numpy as np
import random
import pickle
from torch.utils.data import DataLoader as TorchDataLoader
from sklearn.metrics import r2_score, mean_squared_error

from models import (SetTransformerGraphClassifier,
                    DeepSetGraphClassifier,
                    SetGraphClassifier,
                    SetDataset,
                    collate_sets)

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================================
# Model Wrappers for Regression (1 output instead of num_classes)
# ============================================================================
class SetTransformerRegressor(SetTransformerGraphClassifier):
    def __init__(self, in_channels, hidden_dim):
        super().__init__(in_channels, hidden_dim, num_classes=1)
    
    def forward(self, data, set_batch):
        return super().forward(data, set_batch).squeeze(-1)


class DeepSetRegressor(DeepSetGraphClassifier):
    def __init__(self, in_channels, hidden_dim):
        super().__init__(in_channels, hidden_dim, num_classes=1)
    
    def forward(self, data, set_batch):
        return super().forward(data, set_batch).squeeze(-1)


class GraphSetConvRegressor(SetGraphClassifier):
    def __init__(self, in_channels, hidden_dim):
        super().__init__(in_channels, hidden_dim, num_classes=1)
    
    def forward(self, data, set_batch):
        return super().forward(data, set_batch).squeeze(-1)


def get_model(model_name, in_channels, hidden_dim):
    if model_name == 'SetTransformer':
        return SetTransformerRegressor(in_channels, hidden_dim)
    elif model_name == 'DeepSets':
        return DeepSetRegressor(in_channels, hidden_dim)
    elif model_name == 'GraphSetConv':
        return GraphSetConvRegressor(in_channels, hidden_dim)


def load_buchwald_hartwig(data_dir='/home/josee/Documents/JoseE/Tests/Data/buchwald_hartwig_processed'):
    """Load Buchwald-Hartwig reaction yield dataset."""
    
    def load_split(filename):
        with open(f'{data_dir}/{filename}', 'rb') as f:
            data = pickle.load(f)
        
        sets = []
        for rxn in data:
            reactants = rxn['reactants']
            yield_val = rxn['yield']
            for g in reactants:
                g.x = g.x.float()
            sets.append((reactants, yield_val))
        return sets
    
    train_sets = load_split('train_reactions_graphs.pkl')
    val_sets = load_split('valid_reactions_graphs.pkl')
    test_sets = load_split('test_reactions_graphs.pkl')
    
    return train_sets, val_sets, test_sets


def _align_targets(pred, targets, set_batch):
    """Align targets to pred shape. If pred is per-graph and targets are per-graph,
    use directly. If pred is per-set, aggregate per-graph targets to per-set."""
    if pred.size(0) == targets.size(0):
        return targets
    num_sets = int(set_batch.max()) + 1
    set_targets = torch.zeros(num_sets, dtype=targets.dtype, device=targets.device)
    set_targets.scatter_(0, set_batch, targets)
    return set_targets


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data, set_batch, targets in loader:
        data = data.to(device)
        set_batch = set_batch.to(device)
        targets = targets.to(device).float()

        optimizer.zero_grad()
        pred = model(data, set_batch)
        targets = _align_targets(pred, targets, set_batch)
        loss = F.mse_loss(pred, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, set_batch, targets in loader:
            data = data.to(device)
            set_batch = set_batch.to(device)
            targets = targets.to(device)
            pred = model(data, set_batch)
            targets = _align_targets(pred, targets, set_batch)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    r2 = r2_score(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    return r2, rmse


def main():
   
    device = torch.device('cuda')
    print(f"Using device: {device}")

    # Parameters
    model_names = ['SetTransformer', 'DeepSets', 'GraphSetConv']
    num_epochs = 100
    hidden_dim = 64
    batch_size = 32
    learning_rate = 1e-3

    all_results = {name: {'train_loss': [], 'val_r2': [], 'val_rmse': []} 
                   for name in model_names}

    # Load Buchwald-Hartwig dataset
    print("Loading Buchwald-Hartwig dataset...")
    train_sets, val_sets, test_sets = load_buchwald_hartwig()
    print(f"Train: {len(train_sets)}, Val: {len(val_sets)}, Test: {len(test_sets)}")

    # Get input dimensions from first graph
    in_channels = train_sets[0][0][0].x.shape[1]  # First reaction, first reactant, features
    print(f"Input channels: {in_channels}")
    
    # Debug: Check yield distribution
    train_yields = [s[1] for s in train_sets]
    print(f"\nYield statistics:")
    print(f"  Min: {min(train_yields):.6f}, Max: {max(train_yields):.6f}")
    print(f"  Mean: {np.mean(train_yields):.6f}, Std: {np.std(train_yields):.6f}")
    print(f"  First 5 yields: {train_yields[:5]}")

    # Create DataLoaders
    train_loader = TorchDataLoader(
        SetDataset(train_sets), 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_sets
    )
    val_loader = TorchDataLoader(
        SetDataset(val_sets), 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_sets
    )
    test_loader = TorchDataLoader(
        SetDataset(test_sets), 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_sets
    )

    # Train each model
    for model_name in model_names:
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")
        
        model = get_model(model_name, in_channels, hidden_dim)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        best_val_r2 = -float('inf')
        for epoch in range(num_epochs):
            train_loss = train_epoch(model, train_loader, optimizer, device)
            val_r2, val_rmse = evaluate(model, val_loader, device)

            all_results[model_name]['train_loss'].append(train_loss)
            all_results[model_name]['val_r2'].append(val_r2)
            all_results[model_name]['val_rmse'].append(val_rmse)

            if val_r2 > best_val_r2:
                best_val_r2 = val_r2

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.8f}, Val R²: {val_r2:.4f}, Val RMSE: {val_rmse:.6f}")

        # Test evaluation with debug output
        test_r2, test_rmse = evaluate(model, test_loader, device)
        print(f"Best Val R² for {model_name}: {best_val_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}, Test RMSE: {test_rmse:.6f}")
        
        # Debug: Show sample predictions
        model.eval()
        with torch.no_grad():
            sample_data, sample_batch, sample_targets = next(iter(test_loader))
            sample_preds = model(sample_data.to(device), sample_batch.to(device))
            print(f"Sample predictions: {sample_preds[:5].cpu().numpy()}")
            print(f"Sample targets:     {sample_targets[:5].numpy()}")

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for model_name in model_names:
        axes[0].plot(all_results[model_name]['train_loss'], label=model_name)
        axes[1].plot(all_results[model_name]['val_r2'], label=model_name)
        axes[2].plot(all_results[model_name]['val_rmse'], label=model_name)

    axes[0].set_title('Train Loss (MSE)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].set_title('Validation R²')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('R²')
    axes[1].legend()

    axes[2].set_title('Validation RMSE')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('RMSE')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('buchwald_comparison.png')
    print("\nSaved plot to buchwald_comparison.png")

    # Summary Table
    summary = pd.DataFrame({
        'Model': model_names,
        'Best Val R²': [max(all_results[m]['val_r2']) for m in model_names],
        'Final Val RMSE': [all_results[m]['val_rmse'][-1] for m in model_names],
        'Final Train Loss': [all_results[m]['train_loss'][-1] for m in model_names],
    })
    summary.to_csv('buchwald_comparison.csv', index=False)
    print("\nSaved summary to buchwald_comparison.csv")
    print(summary)


if __name__ == '__main__':
    main()
