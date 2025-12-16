"""
EGNN Data Flow Test Script
===========================
Tests if data is being processed correctly by the EGNN architecture.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from functools import partial

from Data.sdf_dataset import SDFConformerDataset
from Data.collate import collate_molecule_batch
from Models.conformer_set_model import ConformerSetModel
from Models.egnn import EGNN


def test_data_shapes():
    """Test that data has correct shapes."""
    print("=" * 60)
    print("TEST 1: Data Loading and Shapes")
    print("=" * 60)
    
    dataset = SDFConformerDataset(
        sdf_path="Data/datasets/Drugs.sdf",
        csv_path="Data/datasets/Drugs.csv",
        cache_path="Data/datasets/drugs_cache.pkl",
        max_conformers=10
    )
    
    sample = dataset[0]
    print(f"\nSingle sample keys: {sample.keys()}")
    print(f"  conformers: list of {len(sample['conformers'])} conformers")
    print(f"  targets: {sample['targets'].shape} = {sample['targets']}")
    
    conf = sample['conformers'][0]
    print(f"\nConformer keys: {conf.keys()}")
    print(f"  atom_features: {conf['atom_features'].shape}")
    print(f"  positions: {conf['positions'].shape}")
    print(f"  edge_index: {conf['edge_index'].shape}")
    
    return dataset


def test_collate_batch(dataset):
    """Test collate function output."""
    print("\n" + "=" * 60)
    print("TEST 2: Collate Function (Batching)")
    print("=" * 60)
    
    collate_fn = partial(collate_molecule_batch, max_conformers=10)
    
    # Create a small batch
    samples = [dataset[i] for i in range(4)]
    batch = collate_fn(samples)
    
    print(f"\nBatch keys: {batch.keys()}")
    print(f"  targets: {batch['targets'].shape}")
    print(f"  conformer_batches: list of {len(batch['conformer_batches'])} molecules")
    
    for i, conf_batch in enumerate(batch['conformer_batches'][:2]):
        print(f"\n  Molecule {i}:")
        print(f"    x: {conf_batch['x'].shape}")
        print(f"    pos: {conf_batch['pos'].shape}")
        print(f"    adj_mat: {conf_batch['adj_mat'].shape}")
        print(f"    num_conformers: {conf_batch['num_conformers']}")
    
    return batch


def test_egnn_forward(batch):
    """Test EGNN forward pass."""
    print("\n" + "=" * 60)
    print("TEST 3: EGNN Forward Pass")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    egnn = EGNN(
        input_dim=6,
        hidden_dim=128,
        num_layers=4
    ).to(device)
    
    # Test with single molecule's conformers
    conf_batch = batch['conformer_batches'][0]
    x = conf_batch['x'].to(device)
    pos = conf_batch['pos'].to(device)
    adj_mat = conf_batch['adj_mat'].to(device)
    
    print(f"\nInput shapes:")
    print(f"  x: {x.shape}")
    print(f"  pos: {pos.shape}")
    print(f"  adj_mat: {adj_mat.shape}")
    
    # Forward pass
    embeddings = egnn(x, pos, adj_mat)
    print(f"\nOutput embedding: {embeddings.shape}")
    print(f"  Mean: {embeddings.mean().item():.4f}")
    print(f"  Std: {embeddings.std().item():.4f}")
    print(f"  Min: {embeddings.min().item():.4f}")
    print(f"  Max: {embeddings.max().item():.4f}")
    
    return egnn


def test_gradient_flow(batch):
    """Test if gradients flow through EGNN."""
    print("\n" + "=" * 60)
    print("TEST 4: Gradient Flow Test")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ConformerSetModel(
        input_dim=6,
        hidden_dim=128,
        output_dim=128,
        num_targets=1,
        num_egnn_layers=4,
        use_conformer_mha=True,
        num_mha_heads=4,
        num_mha_layers=1
    ).to(device)
    
    # Move targets to device (model uses this to detect device)
    batch['targets'] = batch['targets'].to(device)
    
    # Forward pass
    predictions = model(batch)
    targets = batch['targets'][:, 0:1]
    
    print(f"\nPredictions: {predictions.shape}")
    print(f"Targets: {targets.shape}")
    
    # Compute loss and backward
    loss = nn.MSELoss()(predictions, targets)
    loss.backward()
    
    print(f"\nLoss: {loss.item():.4f}")
    
    # Check gradients
    print("\nGradient norms by component:")
    grad_summary = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            component = name.split('.')[0]
            if component not in grad_summary:
                grad_summary[component] = []
            grad_summary[component].append(grad_norm)
    
    for component, norms in grad_summary.items():
        avg_norm = sum(norms) / len(norms)
        print(f"  {component}: avg={avg_norm:.6f}, max={max(norms):.6f}, min={min(norms):.6f}")
    
    # Check specifically edge_mlp
    print("\nEGNN layer details:")
    for name, param in model.named_parameters():
        if 'egnn' in name and param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm < 0.0001:
                print(f"  ⚠️  {name}: grad_norm={grad_norm:.8f} (VERY LOW)")
            else:
                print(f"  ✓  {name}: grad_norm={grad_norm:.6f}")


def test_adjacency_matrix(batch):
    """Test adjacency matrix properties."""
    print("\n" + "=" * 60)
    print("TEST 5: Adjacency Matrix Analysis")
    print("=" * 60)
    
    conf_batch = batch['conformer_batches'][0]
    adj_mat = conf_batch['adj_mat']
    
    print(f"\nAdjacency matrix shape: {adj_mat.shape}")
    print(f"  Non-zero elements: {(adj_mat > 0).sum().item()}")
    print(f"  Total elements: {adj_mat.numel()}")
    print(f"  Sparsity: {1 - (adj_mat > 0).sum().item() / adj_mat.numel():.2%}")
    
    # Check if symmetric
    is_symmetric = torch.allclose(adj_mat, adj_mat.transpose(-1, -2))
    print(f"  Symmetric: {is_symmetric}")
    
    # Check average edges per atom
    avg_edges = (adj_mat > 0).sum(dim=-1).float().mean().item()
    print(f"  Avg edges per atom: {avg_edges:.1f}")


def test_position_variance(batch):
    """Test if conformer positions vary."""
    print("\n" + "=" * 60)
    print("TEST 6: Conformer Position Variance")
    print("=" * 60)
    
    conf_batch = batch['conformer_batches'][0]
    pos = conf_batch['pos']  # [num_conf, num_atoms, 3]
    
    print(f"\nPositions shape: {pos.shape}")
    
    # Compute variance across conformers for each atom
    pos_variance = pos.var(dim=0).mean()
    print(f"  Mean position variance across conformers: {pos_variance:.4f}")
    
    # Check if positions are actually different
    if pos.shape[0] > 1:
        pos_diff = (pos[0] - pos[1]).abs().mean()
        print(f"  Mean position difference (conf 0 vs 1): {pos_diff:.4f}")
    
    # Check distance distribution
    diff = pos.unsqueeze(2) - pos.unsqueeze(1)  # [C, N, N, 3]
    distances = torch.norm(diff, dim=-1)  # [C, N, N]
    
    print(f"  Distance stats:")
    print(f"    Mean: {distances.mean().item():.2f} Å")
    print(f"    Max: {distances.max().item():.2f} Å")
    print(f"    Min (non-zero): {distances[distances > 0].min().item():.2f} Å")


def main():
    print("\n🔍 EGNN Data Flow Diagnostic Test\n")
    
    # Run all tests
    dataset = test_data_shapes()
    batch = test_collate_batch(dataset)
    test_egnn_forward(batch)
    test_gradient_flow(batch)
    test_adjacency_matrix(batch)
    test_position_variance(batch)
    
    print("\n" + "=" * 60)
    print("✓ All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
