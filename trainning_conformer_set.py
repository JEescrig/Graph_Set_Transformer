"""
Training Script for ConformerSetModel
=====================================
Multi-target regression on Drugs dataset (energy, ip, ea, chi)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm
import numpy as np

from Data.sdf_dataset import SDFConformerDataset
from Data.collate import collate_molecule_batch
from Models.conformer_set_model import ConformerSetModel


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training"):
        # Move targets to device (conformer_batches are moved inside model)
        batch['targets'] = batch['targets'].to(device)
        
        # Forward
        optimizer.zero_grad()
        predictions = model(batch)
        loss = criterion(predictions, batch['targets'])
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            batch['targets'] = batch['targets'].to(device)
            
            predictions = model(batch)
            loss = criterion(predictions, batch['targets'])
            
            total_loss += loss.item()
            all_preds.append(predictions.cpu())
            all_targets.append(batch['targets'].cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Per-target MAE
    mae_per_target = torch.abs(all_preds - all_targets).mean(dim=0)
    
    return total_loss / len(loader), mae_per_target


def main():
    # ============================================================
    # Configuration
    # ============================================================
    BATCH_SIZE = 32
    MAX_CONFORMERS = 10
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 5  # Short test
    HIDDEN_DIM = 128
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ============================================================
    # Load Data (with caching to avoid reparsing SDF every time)
    # ============================================================
    print("\nLoading dataset...")
    
    CACHE_PATH = "Data/datasets/drugs_cache.pkl"
    
    # Load full dataset (will use cache if exists)
    full_dataset = SDFConformerDataset(
        sdf_path="Data/datasets/Drugs.sdf",
        csv_path="Data/datasets/Drugs.csv",
        cache_path=CACHE_PATH,
        max_conformers=MAX_CONFORMERS
    )
    
    # Create train/val/test splits
    n_molecules = len(full_dataset)
    np.random.seed(42)
    indices = np.random.permutation(n_molecules)
    
    n_train = int(n_molecules * 0.8)
    n_val = int(n_molecules * 0.1)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    from torch.utils.data import Subset
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    # Create DataLoaders
    collate_fn = partial(collate_molecule_batch, max_conformers=MAX_CONFORMERS)
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # ============================================================
    # Initialize Model
    # ============================================================
    model = ConformerSetModel(
        input_dim=6,  # From MoleculePreprocessor
        hidden_dim=HIDDEN_DIM,
        output_dim=HIDDEN_DIM,
        num_targets=4,  # energy, ip, ea, chi
        num_egnn_layers=3,
        use_conformer_mha=True,
        num_mha_heads=4,
        num_mha_layers=1
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # ============================================================
    # Training Setup
    # ============================================================
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )
    
    best_val_loss = float('inf')
    target_names = ['energy', 'ip', 'ea', 'chi']
    
    # ============================================================
    # Training Loop
    # ============================================================
    print("\nStarting training...")
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 50)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_mae = evaluate(model, val_loader, criterion, device)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val MAE per target:")
        for name, mae in zip(target_names, val_mae):
            print(f"  {name}: {mae:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_conformer_model.pth')
            print("✓ Saved best model")
    
    # ============================================================
    # Test on Best Model
    # ============================================================
    print("\n" + "=" * 50)
    print("Testing best model...")
    print("=" * 50)
    
    model.load_state_dict(torch.load('best_conformer_model.pth'))
    test_loss, test_mae = evaluate(model, test_loader, criterion, device)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test MAE per target:")
    for name, mae in zip(target_names, test_mae):
        print(f"  {name}: {mae:.4f}")


if __name__ == "__main__":
    main()
