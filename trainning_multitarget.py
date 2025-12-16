"""
Training Script for ConformerSetModel - All Targets
====================================================
Multi-target regression: energy, ip, ea, chi
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from functools import partial
from tqdm import tqdm
import numpy as np

from Data.sdf_dataset import SDFConformerDataset
from Data.collate import collate_molecule_batch
from Models.conformer_set_model import ConformerSetModel


# Per-target scale factors (to bring all targets to similar magnitude)
# energy: ~-46000, ip: ~6, ea: ~2.4, chi: ~4.3
TARGET_SCALES = torch.tensor([1.0, 1.0, 1.0])
TARGET_NAMES = ['ip', 'ea', 'chi']


def train_epoch(model, loader, optimizer, criterion, device, target_scales, scaler=None):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        batch['targets'] = batch['targets'].to(device)
        
        # Scale all targets
        targets = batch['targets'][:, 1:] / target_scales.to(device) 
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                predictions = model(batch)
                loss = criterion(predictions, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            predictions = model(batch)
            loss = criterion(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device, target_scales):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            batch['targets'] = batch['targets'].to(device)
            targets_scaled = batch['targets'][:, 1:] / target_scales.to(device)
            
            predictions = model(batch)
            loss = criterion(predictions, targets_scaled)
            
            # Convert back to original scale
            predictions_orig = predictions * target_scales.to(device)
            
            total_loss += loss.item()
            all_preds.append(predictions_orig.cpu())
            all_targets.append(batch['targets'][:, 1:].cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Per-target metrics
    metrics = {}
    for i, name in enumerate(TARGET_NAMES):
        pred = all_preds[:, i]
        target = all_targets[:, i]
        
        mae = torch.abs(pred - target).mean().item()
        ss_res = ((target - pred) ** 2).sum()
        ss_tot = ((target - target.mean()) ** 2).sum()
        r2 = (1 - ss_res / ss_tot).item()
        
        metrics[name] = {'mae': mae, 'r2': r2}
    
    # Average R²
    avg_r2 = np.mean([m['r2'] for m in metrics.values()])
    
    return total_loss / len(loader), metrics, avg_r2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mixed-precision', action='store_true')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ============================================================
    # Load Data
    # ============================================================
    print("\nLoading dataset...")
    
    CACHE_PATH = "Data/datasets/drugs_cache.pkl"
    MAX_CONFORMERS = 20
    
    full_dataset = SDFConformerDataset(
        sdf_path="Data/datasets/Drugs.sdf",
        csv_path="Data/datasets/Drugs.csv",
        cache_path=CACHE_PATH,
        max_conformers=MAX_CONFORMERS
    )
    
    # Create splits
    n_molecules = len(full_dataset)
    np.random.seed(42)
    indices = np.random.permutation(n_molecules)
    
    n_train = int(n_molecules * 0.7)
    n_val = int(n_molecules * 0.3)
    
    train_dataset = Subset(full_dataset, indices[:n_train])
    val_dataset = Subset(full_dataset, indices[n_train:n_train + n_val])
    test_dataset = Subset(full_dataset, indices[n_train + n_val:])
    
    collate_fn = partial(collate_molecule_batch, max_conformers=MAX_CONFORMERS)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.workers, pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.workers, pin_memory=False
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"Target scales: {dict(zip(TARGET_NAMES, TARGET_SCALES.tolist()))}")
    
    # ============================================================
    # Initialize Model
    # ============================================================
    model = ConformerSetModel(
        input_dim=9,  # 9 atom features (MARCEL paper Table S1)
        hidden_dim=128,
        output_dim=64,
        num_targets=3, 
        num_egnn_layers=4,
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    scaler = torch.amp.GradScaler('cuda') if args.mixed_precision else None
    
    best_val_loss = float('inf')
    
    # ============================================================
    # Training Loop
    # ============================================================
    print("\nStarting training...")
    print(f"Targets: {TARGET_NAMES}")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 60)
        
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, TARGET_SCALES, scaler
        )
        val_loss, metrics, avg_r2 = evaluate(
            model, val_loader, criterion, device, TARGET_SCALES
        )
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Avg R²: {avg_r2:.4f}")
        print("\nPer-target metrics:")
        for name in TARGET_NAMES:
            m = metrics[name]
            unit = 'kcal/mol' if name == 'energy' else 'eV'
            print(f"  {name:8s}: MAE={m['mae']:8.2f} {unit:10s}  R²={m['r2']:.4f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'target_scales': TARGET_SCALES,
                'target_names': TARGET_NAMES,
            }, 'best_multitarget_model.pth')
            print("✓ Saved best model")
    
    # ============================================================
    # Final Test
    # ============================================================
    print("\n" + "=" * 60)
    print("Testing best model...")
    print("=" * 60)
    
    checkpoint = torch.load('best_multitarget_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, metrics, avg_r2 = evaluate(
        model, test_loader, criterion, device, TARGET_SCALES
    )
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Avg R²: {avg_r2:.4f}")
    print("\nFinal Test Results:")
    for name in TARGET_NAMES:
        m = metrics[name]
        unit = 'kcal/mol' if name == 'energy' else 'eV'
        print(f"  {name:8s}: MAE={m['mae']:8.2f} {unit:10s}  R²={m['r2']:.4f}")


if __name__ == "__main__":
    main()
