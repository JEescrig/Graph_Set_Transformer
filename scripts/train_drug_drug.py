"""
Training script for Drug-Drug Interaction classification.
Multi-class classification using set-based models on drug pairs.
"""
import torch
import torch.nn.functional as F
import numpy as np
import random
import pickle
from torch.utils.data import DataLoader as TorchDataLoader
from sklearn.metrics import accuracy_score, f1_score

from models import (SetTransformerGraphClassifier,
                    DeepSetGraphClassifier,
                    SetGraphClassifier,
                    SetDataset,
                    collate_sets)

import matplotlib.pyplot as plt
import pandas as pd


def get_model(model_name, in_channels, hidden_dim, num_classes):
    if model_name == 'SetTransformer':
        return SetTransformerGraphClassifier(in_channels, hidden_dim, num_classes)
    elif model_name == 'DeepSets':
        return DeepSetGraphClassifier(in_channels, hidden_dim, num_classes)
    elif model_name == 'GraphSetConv':
        return SetGraphClassifier(in_channels, hidden_dim, num_classes)


def load_drug_drug(data_dir='/home/josee/Documents/JoseE/Tests/Data/Drug_Drug'):
    """
    Load Drug-Drug Interaction dataset.
    Returns train, val, test lists of (drug_pair_graphs, label) tuples.
    """
    
    def load_split(filename):
        with open(f'{data_dir}/{filename}', 'rb') as f:
            data = pickle.load(f)
        
        sets = []
        for interaction in data:
            # Combine both drugs as a set (drug1 + drug2)
            drug_pair = interaction['reactants'] + interaction['products']
            label = int(interaction['yield'])  # Class label
            for g in drug_pair:
                g.x = g.x.float()
            sets.append((drug_pair, label))
        return sets
    
    train_sets = load_split('train_reactions_graphs.pkl')
    val_sets = load_split('valid_reactions_graphs.pkl')
    test_sets = load_split('test_reactions_graphs.pkl')
    
    return train_sets, val_sets, test_sets


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data, set_batch, targets in loader:
        data = data.to(device)
        set_batch = set_batch.to(device)
        targets = targets.to(device).long()

        optimizer.zero_grad()
        pred = model(data, set_batch)
        loss = F.cross_entropy(pred, targets)
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
            logits = model(data, set_batch)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    return acc, f1


def main():
    # Force CPU - RTX 5090 (sm_120) requires PyTorch nightly with CUDA 12.8+
    device = torch.device('cuda')
    print(f"Using device: {device}")

    # Parameters
    model_names = ['SetTransformer', 'DeepSets', 'GraphSetConv']
    num_epochs = 50  # Fewer epochs for larger dataset
    hidden_dim = 64
    batch_size = 64
    learning_rate = 1e-3

    all_results = {name: {'train_loss': [], 'val_acc': [], 'val_f1': []} 
                   for name in model_names}

    # Load Drug-Drug dataset
    print("Loading Drug-Drug Interaction dataset...")
    train_sets, val_sets, test_sets = load_drug_drug()
    print(f"Train: {len(train_sets)}, Val: {len(val_sets)}, Test: {len(test_sets)}")

    # Get input dimensions and number of classes
    in_channels = train_sets[0][0][0].x.shape[1]
    all_labels = [s[1] for s in train_sets]
    num_classes = max(all_labels) + 1
    print(f"Input channels: {in_channels}, Num classes: {num_classes}")

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
        
        model = get_model(model_name, in_channels, hidden_dim, num_classes)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        best_val_acc = 0
        for epoch in range(num_epochs):
            train_loss = train_epoch(model, train_loader, optimizer, device)
            val_acc, val_f1 = evaluate(model, val_loader, device)

            all_results[model_name]['train_loss'].append(train_loss)
            all_results[model_name]['val_acc'].append(val_acc)
            all_results[model_name]['val_f1'].append(val_f1)

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        # Test evaluation
        test_acc, test_f1 = evaluate(model, test_loader, device)
        print(f"Best Val Accuracy for {model_name}: {best_val_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for model_name in model_names:
        axes[0].plot(all_results[model_name]['train_loss'], label=model_name)
        axes[1].plot(all_results[model_name]['val_acc'], label=model_name)
        axes[2].plot(all_results[model_name]['val_f1'], label=model_name)

    axes[0].set_title('Train Loss (Cross-Entropy)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].set_title('Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    axes[2].set_title('Validation F1 (Macro)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('drug_drug_comparison.png')
    print("\nSaved plot to drug_drug_comparison.png")

    # Summary Table
    summary = pd.DataFrame({
        'Model': model_names,
        'Best Val Acc': [max(all_results[m]['val_acc']) for m in model_names],
        'Best Val F1': [max(all_results[m]['val_f1']) for m in model_names],
        'Final Train Loss': [all_results[m]['train_loss'][-1] for m in model_names],
    })
    summary.to_csv('drug_drug_comparison.csv', index=False)
    print("\nSaved summary to drug_drug_comparison.csv")
    print(summary)


if __name__ == '__main__':
    main()
