"""
Training Script for ConformerSetModel - Energy Only
====================================================
Single-target regression on Drugs dataset (energy only)
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


class TargetNormalizer:
    """Normalize targets to zero mean and unit variance."""
    
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, targets):
        self.mean = targets.mean(dim=0, keepdim=True)
        self.std = targets.std(dim=0, keepdim=True)
        self.std[self.std < 1e-6] = 1.0
        return self
    
    def transform(self, targets):
        return (targets - self.mean) / self.std
    
    def inverse_transform(self, normalized):
        return normalized * self.std + self.mean
    Testing best model...
==================================================
Evaluating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 235/235 [00:21<00:00, 10.68it/s]

Test Loss: 0.0093
Test MAE (energy): 727.52 kcal/mol
Test R²: 0.9902

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


def train_epoch(model, loader, optimizer, criterion, device, normalizer=None, scaler=None, scale_factor=1.0):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        # Move targets to device first (model uses this to detect device)
        batch['targets'] = batch['targets'].to(device)
        
        # Extract only energy (first target)
        targets = batch['targets'][:, 0:1]
        
        if normalizer:
            targets = normalizer.transform(targets)
        else:
            # Simple scaling when not using normalizer
            targets = targets / scale_factor
        
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
            
            # Debug: Check gradient norms on first batch
            if total_loss == 0:  # First batch
                total_grad = 0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        total_grad += grad_norm
                        if 'egnn' in name and grad_norm > 0:
                            print(f"  {name}: grad_norm={grad_norm:.6f}")
                print(f"Total gradient norm: {total_grad:.6f}")
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device, normalizer=None, scale_factor=1.0):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            # Move targets to device first
            batch['targets'] = batch['targets'].to(device)
            targets = batch['targets'][:, 0:1]
            
            if normalizer:
                targets_norm = normalizer.transform(targets)
            else:
                targets_norm = targets / scale_factor
            
            predictions = model(batch)
            loss = criterion(predictions, targets_norm)
            
            # Convert predictions back to original scale
            if normalizer:
                predictions = normalizer.inverse_transform(predictions)
            else:
                predictions = predictions * scale_factor
            
            total_loss += loss.item()
            all_preds.append(predictions.cpu())
            all_targets.append(batch['targets'][:, 0:1].cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    mae = torch.abs(all_preds - all_targets).mean()
    
    # R² score
    ss_res = ((all_targets - all_preds) ** 2).sum()
    ss_tot = ((all_targets - all_targets.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot)
    
    return total_loss / len(loader), mae, r2.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mixed-precision', action='store_true')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)  # Balanced LR
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--no-normalize', action='store_true', help='Disable target normalization')
    parser.add_argument('--scale-factor', type=float, default=10000.0, help='Scale targets by this factor')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ============================================================
    # Load Data
    # ============================================================
    print("\nLoading dataset...")
    
    CACHE_PATH = "Data/datasets/drugs_cache.pkl"
    MAX_CONFORMERS = 10
    
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
    
    n_train = int(n_molecules * 0.8)
    n_val = int(n_molecules * 0.1)
    
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
    
    # ============================================================
    # Target Normalization/Scaling
    # ============================================================
    normalizer = None
    scale_factor = args.scale_factor if args.no_normalize else 1.0
    
    if not args.no_normalize:
        print("Computing target normalization for energy...")
        
        all_targets = []
        for i in range(len(train_dataset)):
            sample = train_dataset[i]
            all_targets.append(sample['targets'][0:1])  # Only energy
        all_targets = torch.stack(all_targets)
        
        normalizer = TargetNormalizer().fit(all_targets).to(device)
        
        print(f"Energy mean: {normalizer.mean.item():.2f}")
        print(f"Energy std: {normalizer.std.item():.2f}")
    else:
        print(f"Training WITHOUT normalization, scaling by {scale_factor}")
    
    # ============================================================
    # Initialize Model
    # ============================================================
    model = ConformerSetModel(
        input_dim=6,
        hidden_dim=256,
        output_dim=128,
        num_targets=1,  # Only energy
        num_egnn_layers=4,
        use_conformer_mha=True,  # Simpler, like MARCEL baseline (DeepSets only)
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
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, normalizer, scaler, scale_factor
        )
        val_loss, val_mae, val_r2 = evaluate(model, val_loader, criterion, device, normalizer, scale_factor)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val MAE (energy): {val_mae:.2f} kcal/mol")
        print(f"Val R²: {val_r2:.4f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dict = {
                'model_state_dict': model.state_dict(),
                'scale_factor': scale_factor,
            }
            if normalizer is not None:
                save_dict['normalizer_mean'] = normalizer.mean
                save_dict['normalizer_std'] = normalizer.std
            torch.save(save_dict, 'best_energy_model.pth')
            print("✓ Saved best model")
    
    # ============================================================
    # Final Test
    # ============================================================
    print("\n" + "=" * 50)
    print("Testing best model...")
    print("=" * 50)
    
    checkpoint = torch.load('best_energy_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_mae, test_r2 = evaluate(model, test_loader, criterion, device, normalizer, scale_factor)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test MAE (energy): {test_mae:.2f} kcal/mol")
    print(f"Test R²: {test_r2:.4f}")


if __name__ == "__main__":
    main()
