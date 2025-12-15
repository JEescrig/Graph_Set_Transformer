"""
Test script for the Drugs dataset pipeline.
Verifies that the SDF parser, dataset, and collate function work correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader
from functools import partial

from sdf_dataset import SDFConformerDataset
from collate import collate_conformer_batch


def test_pipeline():
    print("=" * 60)
    print("DRUGS DATASET PIPELINE TEST")
    print("=" * 60)
    
    sdf_path = "datasets/Drugs.sdf"
    csv_path = "datasets/Drugs.csv"
    
    if not os.path.exists(sdf_path):
        print(f"❌ SDF file not found: {sdf_path}")
        return False
    
    # Test 1: Load a small subset
    print("\n[1] Testing SDFConformerDataset...")
    try:
        dataset = SDFConformerDataset(
            sdf_path=sdf_path,
            csv_path=csv_path,
            max_conformers=5,
            use_sdf_properties=True
        )
        print(f"   ✓ Loaded {len(dataset)} molecules")
    except Exception as e:
        print(f"   ❌ Failed to load dataset: {e}")
        return False
    
    # Test 2: Get a sample
    print("\n[2] Testing __getitem__...")
    try:
        sample = dataset[0]
        print(f"   ✓ Molecule: {sample['molecule_name']}")
        print(f"   ✓ Conformers: {len(sample['conformers'])}")
        print(f"   ✓ Targets shape: {sample['targets'].shape}")
        print(f"   ✓ Target values: {sample['targets']}")
    except Exception as e:
        print(f"   ❌ Failed to get sample: {e}")
        return False
    
    # Test 3: DataLoader with collate function
    print("\n[3] Testing DataLoader with collate...")
    try:
        collate_fn = partial(collate_conformer_batch, max_conformers=5)
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_fn
        )
        
        batch = next(iter(loader))
        print(f"   ✓ Batch size: {batch['batch_size']}")
        print(f"   ✓ Num conformers: {batch['num_conformers']}")
        print(f"   ✓ Graph batch: {batch['graph']}")
        print(f"   ✓ Targets shape: {batch['targets'].shape}")
        print(f"   ✓ Conformer-to-molecule map: {batch['conformer_to_molecule'][:10]}...")
    except Exception as e:
        print(f"   ❌ Failed to batch data: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Verify data shapes for model input
    print("\n[4] Verifying data shapes...")
    try:
        graph = batch['graph']
        print(f"   ✓ Node features (x): {graph.x.shape}")
        print(f"   ✓ Positions (pos): {graph.pos.shape}")
        print(f"   ✓ Edge index: {graph.edge_index.shape}")
        print(f"   ✓ Batch vector: {graph.batch.shape}")
        
        # Verify no NaNs
        assert not torch.isnan(graph.x).any(), "NaNs in node features"
        assert not torch.isnan(graph.pos).any(), "NaNs in positions"
        assert not torch.isnan(batch['targets']).any(), "NaNs in targets"
        print(f"   ✓ No NaN values detected")
    except Exception as e:
        print(f"   ❌ Data validation failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
