"""
Parallel Training Script for ConformerSetModel
===============================================
Multi-GPU training with target normalization.

Usage:
    Single GPU:  python trainning_parallel.py
    Multi-GPU:   python trainning_parallel.py --multi-gpu
    Distributed: torchrun --nproc_per_node=2 trainning_parallel.py --distributed
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
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
        """Compute mean and std from training targets."""
        self.mean = targets.mean(dim=0)
        self.std = targets.std(dim=0)
        self.std[self.std < 1e-6] = 1.0  # Avoid division by zero
        return self
    
    def transform(self, targets):
        """Normalize targets."""
        return (targets - self.mean) / self.std
    
    def inverse_transform(self, normalized):
        """Convert back to original scale."""
        return normalized * self.std + self.mean
    
    def to(self, device):
        """Move normalizer to device."""
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


def train_epoch(model, loader, optimizer, criterion, device, normalizer=None, scaler=None):
    """Train for one epoch with optional mixed precision."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc="Training", disable=not is_main_process())
    for batch in pbar:
        batch['targets'] = batch['targets'].to(device)
        targets = batch['targets']
        
        if normalizer:
            targets = normalizer.transform(targets)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
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


def evaluate(model, loader, criterion, device, normalizer=None):
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(loader, desc="Evaluating", disable=not is_main_process())
    with torch.no_grad():
        for batch in pbar:
            batch['targets'] = batch['targets'].to(device)
            targets = batch['targets']
            
            if normalizer:
                targets_norm = normalizer.transform(targets)
            else:
                targets_norm = targets
            
            predictions = model(batch)
            loss = criterion(predictions, targets_norm)
            
            # Convert predictions back to original scale for MAE
            if normalizer:
                predictions = normalizer.inverse_transform(predictions)
            
            total_loss += loss.item()
            all_preds.append(predictions.cpu())
            all_targets.append(batch['targets'].cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    mae_per_target = torch.abs(all_preds - all_targets).mean(dim=0)
    
    return total_loss / len(loader), mae_per_target


def is_main_process():
    """Check if this is the main process (for logging)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def setup_distributed():
    """Initialize distributed training."""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--multi-gpu', action='store_true', help='Use DataParallel for multi-GPU')
    parser.add_argument('--distributed', action='store_true', help='Use DistributedDataParallel')
    parser.add_argument('--mixed-precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--normalize', action='store_true', default=True, help='Normalize targets')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--workers', type=int, default=4, help='DataLoader workers')
    args = parser.parse_args()
    
    # ============================================================
    # Setup Device
    # ============================================================
    if args.distributed:
        local_rank = setup_distributed()
        device = torch.device(f'cuda:{local_rank}')
    elif args.multi_gpu and torch.cuda.device_count() > 1:
        device = torch.device('cuda:0')
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if is_main_process():
        print(f"Using device: {device}")
    
    # ============================================================
    # Load Data
    # ============================================================
    CACHE_PATH = "Data/datasets/drugs_cache.pkl"
    MAX_CONFORMERS = 10
    
    if is_main_process():
        print("\nLoading dataset...")
    
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
    
    # Collate function
    collate_fn = partial(collate_molecule_batch, max_conformers=MAX_CONFORMERS)
    
    # Samplers for distributed training
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=args.workers,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=args.workers,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.workers,
        pin_memory=False
    )
    
    if is_main_process():
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # ============================================================
    # Target Normalization
    # ============================================================
    normalizer = None
    if args.normalize:
        if is_main_process():
            print("Computing target normalization statistics...")
        
        # Collect all training targets
        all_targets = []
        for i in range(len(train_dataset)):
            sample = train_dataset[i]
            all_targets.append(sample['targets'])
        all_targets = torch.stack(all_targets)
        
        normalizer = TargetNormalizer().fit(all_targets).to(device)
        
        if is_main_process():
            print(f"Target means: {normalizer.mean}")
            print(f"Target stds: {normalizer.std}")
    
    # ============================================================
    # Initialize Model
    # ============================================================
    model = ConformerSetModel(
        input_dim=6,
        hidden_dim=256,
        output_dim=256,
        num_targets=4,
        num_egnn_layers=4,
        use_conformer_mha=True,
        num_mha_heads=4,
        num_mha_layers=1
    )
    
    # Wrap model for parallel training
    if args.distributed:
        model = model.to(device)
        model = DistributedDataParallel(model, device_ids=[local_rank])
    elif args.multi_gpu and torch.cuda.device_count() > 1:
        model = DataParallel(model)
        model = model.to(device)
    else:
        model = model.to(device)
    
    if is_main_process():
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,}")
    
    # ============================================================
    # Training Setup
    # ============================================================
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )
    
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
    
    best_val_loss = float('inf')
    target_names = ['energy', 'ip', 'ea', 'chi']
    
    # ============================================================
    # Training Loop
    # ============================================================
    if is_main_process():
        print("\nStarting training...")
    
    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        if is_main_process():
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print("-" * 50)
        
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, normalizer, scaler
        )
        val_loss, val_mae = evaluate(model, val_loader, criterion, device, normalizer)
        
        if is_main_process():
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val MAE per target:")
            for name, mae in zip(target_names, val_mae):
                print(f"  {name}: {mae:.4f}")
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                torch.save({
                    'model_state_dict': state_dict,
                    'normalizer_mean': normalizer.mean if normalizer else None,
                    'normalizer_std': normalizer.std if normalizer else None,
                }, 'best_conformer_model_parallel.pth')
                print("✓ Saved best model")
    
    # ============================================================
    # Final Test
    # ============================================================
    if is_main_process():
        print("\n" + "=" * 50)
        print("Testing best model...")
        print("=" * 50)
        
        checkpoint = torch.load('best_conformer_model_parallel.pth')
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        test_loss, test_mae = evaluate(model, test_loader, criterion, device, normalizer)
        
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test MAE per target:")
        for name, mae in zip(target_names, test_mae):
            print(f"  {name}: {mae:.4f}")
    
    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
