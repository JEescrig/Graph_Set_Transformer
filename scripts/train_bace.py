"""
BACE Classification Training with SetGCN
Binary classification of BACE inhibitor activity from molecular graphs.
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
import pickle
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report
)

from Set_GCN import SetGCN
from Set_GAT import SetGAT
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


# ============================================================================
# DATASET
# ============================================================================

class MoleculeDataset(Dataset):
    """Dataset for single-molecule classification"""
    
    def __init__(self, graphs):
        self.graphs = graphs
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]


def collate_molecules(batch_list):
    """Collate single molecules into a batch"""
    batched = Batch.from_data_list(batch_list)
    labels = torch.cat([g.y for g in batch_list], dim=0)
    
    # For single molecules, set_batch = batch (each molecule is its own "set")
    num_graphs = len(batch_list)
    set_batch = torch.arange(num_graphs, dtype=torch.long)
    
    return {
        'x': batched.x,
        'edge_index': batched.edge_index,
        'batch': batched.batch,
        'set_batch': set_batch,
        'labels': labels,
    }


# ============================================================================
# CLASSIFIER MODEL
# ============================================================================

class BACEClassifier(nn.Module):
    """Binary classifier for BACE inhibitor activity"""
    
    def __init__(self, in_channels, out_channels=32, num_layers=3):
        super().__init__()
        
        self.encoder = SetGCN(
            in_channels=in_channels,
            hidden_channels=out_channels * 2,
            out_channels=out_channels,
            num_layers=num_layers,
            mha_dropout=0.2,
            ffn_dropout=0.2,
            pooling="mean",
            use_gating=True,
            activation="relu",
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(out_channels, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1) 
        )
    
    def forward(self, x, edge_index, batch, set_batch):
        embeddings = self.encoder(x, edge_index, batch, set_batch)
        logits = self.classifier(embeddings)
        return logits.squeeze(-1)


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_samples = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        x = batch['x'].to(device)
        edge_index = batch['edge_index'].to(device)
        graph_batch = batch['batch'].to(device)
        set_batch = batch['set_batch'].to(device)
        labels = batch['labels'].float().to(device)
        
        optimizer.zero_grad()
        logits = model(x, edge_index, graph_batch, set_batch)
        loss = criterion(logits, labels)
        
        if torch.isnan(loss):
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(labels)
        total_samples += len(labels)
    
    return total_loss / total_samples if total_samples > 0 else 0


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            x = batch['x'].to(device)
            edge_index = batch['edge_index'].to(device)
            graph_batch = batch['batch'].to(device)
            set_batch = batch['set_batch'].to(device)
            labels = batch['labels'].float().to(device)
            
            logits = model(x, edge_index, graph_batch, set_batch)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    metrics = {
        'loss': total_loss / len(all_labels),
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'auroc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0,
    }
    
    return metrics, all_preds, all_labels


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_curves(history, save_path='training_curves.png'):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    axes[0, 0].plot(history['epoch'], history['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(history['epoch'], history['val_loss'], label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history['epoch'], history['val_acc'], linewidth=2, color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[1, 0].plot(history['epoch'], history['val_f1'], linewidth=2, color='orange')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Validation F1')
    axes[1, 0].grid(True, alpha=0.3)
    
    # AUROC
    axes[1, 1].plot(history['epoch'], history['val_auroc'], linewidth=2, color='purple')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AUROC')
    axes[1, 1].set_title('Validation AUROC')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n✓ Saved training curves: {save_path}")


def save_results_table(history, test_metrics, save_path='results'):
    """Save training history and test results"""
    # Training history
    df_history = pd.DataFrame(history)
    df_history.to_csv(f'{save_path}_history.csv', index=False)
    
    # Summary table
    summary = pd.DataFrame({
        'Metric': ['Best Val AUROC', 'Best Val F1', 'Test Accuracy', 'Test F1', 'Test AUROC', 'Test Precision', 'Test Recall'],
        'Value': [
            f"{max(history['val_auroc']):.4f}",
            f"{max(history['val_f1']):.4f}",
            f"{test_metrics['accuracy']:.4f}",
            f"{test_metrics['f1']:.4f}",
            f"{test_metrics['auroc']:.4f}",
            f"{test_metrics['precision']:.4f}",
            f"{test_metrics['recall']:.4f}",
        ]
    })
    summary.to_csv(f'{save_path}_summary.csv', index=False)
    
    print(f"✓ Saved history: {save_path}_history.csv")
    print(f"✓ Saved summary: {save_path}_summary.csv")
    
    # Print summary table
    print("\n" + "="*40)
    print("RESULTS SUMMARY")
    print("="*40)
    print(summary.to_string(index=False))


# ============================================================================
# MAIN
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Parameters
    batch_size = 64
    num_epochs = 500
    learning_rate = 1e-4
    
    # History for tracking
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_auroc': []}
    
    # Create output directory with timestamp
    run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).parent / 'results' / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load data
    data_dir = Path(__file__).parent.parent.parent / 'Data/Bace/processed'
    
    print(f"\nLoading data from: {data_dir}")
    
    with open(data_dir / 'train_graphs.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open(data_dir / 'valid_graphs.pkl', 'rb') as f:
        val_data = pickle.load(f)
    with open(data_dir / 'test_graphs.pkl', 'rb') as f:
        test_data = pickle.load(f)
    
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    
    # Class distribution
    train_pos = sum(1 for g in train_data if g.y.item() == 1)
    print(f"Train positive: {train_pos}/{len(train_data)} ({100*train_pos/len(train_data):.1f}%)")
    
    # Dataloaders
    train_loader = DataLoader(
        MoleculeDataset(train_data),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_molecules
    )
    val_loader = DataLoader(
        MoleculeDataset(val_data),
        batch_size=batch_size,
        collate_fn=collate_molecules
    )
    test_loader = DataLoader(
        MoleculeDataset(test_data),
        batch_size=batch_size,
        collate_fn=collate_molecules
    )
    
    # Model
    in_channels = train_data[0].x.shape[1]
    print(f"\nInput features: {in_channels}")
    
    model = BACEClassifier(
        in_channels=in_channels,
        out_channels=64,
        num_layers=6
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Loss with class weights for imbalance
    pos_weight = torch.tensor([len(train_data) / train_pos - 1]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=10, factor=0.5, verbose=True
    )
    
    # Training loop
    print("\n" + "="*60)
    print("Starting BACE Classification Training")
    print("="*60)
    
    best_val_auroc = 0
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 60)
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | "
              f"F1: {val_metrics['f1']:.4f} | AUROC: {val_metrics['auroc']:.4f}")
        
        # Track history
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_auroc'].append(val_metrics['auroc'])
        
        scheduler.step(val_metrics['auroc'])
        
        if val_metrics['auroc'] > best_val_auroc:
            best_val_auroc = val_metrics['auroc']
            torch.save(model.state_dict(), output_dir / 'best_bace_classifier.pth')
            print("✓ Saved best model")
    
    # Plot training curves
    plot_training_curves(history, save_path=output_dir / 'training_curves.png')
    
    # Test
    print("\n" + "="*60)
    print("Testing best model...")
    print("="*60)
    
    model.load_state_dict(torch.load(output_dir / 'best_bace_classifier.pth'))
    test_metrics, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  AUROC: {test_metrics['auroc']:.4f}")
    
    print("\n" + "="*60)
    print("Classification Report:")
    print("="*60)
    print(classification_report(test_labels, test_preds, target_names=['Inactive', 'Active']))
    
    # Save results
    save_results_table(history, test_metrics, save_path=output_dir / 'results')


if __name__ == "__main__":
    main()
