"""
SDF Conformer Dataset
=====================
Parses conformers from SDF files and creates a PyTorch Dataset
for multi-target molecular property prediction.

Supports caching to avoid reparsing the SDF file on each run.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import pickle
import os
from rdkit import Chem
from collections import defaultdict
from typing import Optional, List, Dict
from tqdm import tqdm

from .preprocessing import MoleculePreprocessor


class SDFConformerDataset(Dataset):
    """
    Dataset that loads conformer ensembles from an SDF file.
    
    Each molecule may have multiple conformers stored as separate
    SDF entries. This dataset groups them by molecule name and
    returns conformer ensembles for each molecule.
    
    Supports caching to avoid reparsing the SDF file on each run.
    """
    
    TARGET_COLUMNS = ['energy', 'ip', 'ea', 'chi']
    
    def __init__(
        self,
        sdf_path: str,
        csv_path: Optional[str] = None,
        cache_path: Optional[str] = None,
        max_conformers: int = 20,
        max_atomic_num: int = 100,
        add_radius_edges: bool = True,
        radius_cutoff: float = 5.0,
        use_sdf_properties: bool = True,
        molecule_indices: Optional[List[int]] = None,
        force_reparse: bool = False
    ):
        """
        Args:
            sdf_path: Path to SDF file with conformers
            csv_path: Optional path to CSV with labels
            cache_path: Optional path to save/load parsed data (e.g., 'drugs_cache.pkl')
            max_conformers: Maximum conformers per molecule
            max_atomic_num: Maximum atomic number for encoding
            add_radius_edges: Whether to add distance-based edges
            radius_cutoff: Cutoff for radius edges (Angstroms)
            use_sdf_properties: If True, read targets from SDF properties
            molecule_indices: Optional list of molecule indices to use (for splits)
            force_reparse: If True, reparse SDF even if cache exists
        """
        self.sdf_path = sdf_path
        self.csv_path = csv_path
        self.cache_path = cache_path
        self.max_conformers = max_conformers
        self.use_sdf_properties = use_sdf_properties
        
        self.preprocessor = MoleculePreprocessor(
            max_atomic_num=max_atomic_num,
            add_radius_edges=add_radius_edges,
            radius_cutoff=radius_cutoff
        )
        
        # Load labels from CSV if provided
        if csv_path is not None:
            self.labels_df = pd.read_csv(csv_path)
            self.labels_df.set_index('name', inplace=True)
        else:
            self.labels_df = None
        
        # Try to load from cache, otherwise parse SDF
        if cache_path and os.path.exists(cache_path) and not force_reparse:
            print(f"Loading cached data from: {cache_path}")
            self.molecules = self._load_cache(cache_path)
        else:
            self.molecules = self._parse_sdf()
            # Save to cache if path provided
            if cache_path:
                self._save_cache(cache_path)
        
        # Filter by indices if provided (for train/val/test splits)
        if molecule_indices is not None:
            mol_names = list(self.molecules.keys())
            filtered_names = [mol_names[i] for i in molecule_indices if i < len(mol_names)]
            self.molecules = {name: self.molecules[name] for name in filtered_names}
        
        self.mol_names = list(self.molecules.keys())
        print(f"Loaded {len(self.mol_names)} molecules with conformers")
    
    def _save_cache(self, cache_path: str):
        """Save parsed molecules to cache file."""
        print(f"Saving cache to: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(self.molecules, f)
        print(f"Cache saved successfully!")
    
    def _load_cache(self, cache_path: str) -> Dict:
        """Load parsed molecules from cache file."""
        with open(cache_path, 'rb') as f:
            molecules = pickle.load(f)
        return molecules
    
    def _parse_sdf(self) -> Dict:
        """Parse SDF file and group conformers by molecule name."""
        print(f"Parsing SDF file: {self.sdf_path}")
        
        molecules = defaultdict(lambda: {
            'conformers': [],
            'targets': None,
            'smiles': None
        })
        
        supplier = Chem.SDMolSupplier(self.sdf_path, removeHs=False)
        
        for mol in tqdm(supplier, desc="Parsing SDF"):
            if mol is None:
                continue
            
            # Get molecule name (links to CSV)
            mol_name = mol.GetProp('name') if mol.HasProp('name') else None
            if mol_name is None:
                continue
            
            # Get SMILES
            if molecules[mol_name]['smiles'] is None:
                smiles = mol.GetProp('smiles') if mol.HasProp('smiles') else Chem.MolToSmiles(mol)
                molecules[mol_name]['smiles'] = smiles
            
            # Get conformer data
            conf = mol.GetConformer()
            positions = torch.tensor(conf.GetPositions(), dtype=torch.float32)
            
            # Get bonds
            bonds = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                bonds.append((i, j))
            
            # Get atom features
            atom_features = self.preprocessor.get_atom_features(mol)
            
            # Get edge index
            edge_index = self.preprocessor.get_edge_index(bonds, positions)
            
            # Get conformer energy from SDF if available
            conf_energy = float(mol.GetProp('energy')) if mol.HasProp('energy') else 0.0
            
            conformer_data = {
                'atom_features': atom_features,
                'positions': positions,
                'edge_index': edge_index,
                'conformer_energy': conf_energy
            }
            
            molecules[mol_name]['conformers'].append(conformer_data)
            
            # Get targets from SDF properties (use first conformer's values)
            if self.use_sdf_properties and molecules[mol_name]['targets'] is None:
                targets = []
                for target_col in self.TARGET_COLUMNS:
                    if mol.HasProp(target_col):
                        targets.append(float(mol.GetProp(target_col)))
                    else:
                        targets.append(0.0)
                molecules[mol_name]['targets'] = torch.tensor(targets, dtype=torch.float32)
        
        # If using CSV labels, update targets
        if self.labels_df is not None:
            for mol_name in molecules:
                if mol_name in self.labels_df.index:
                    row = self.labels_df.loc[mol_name]
                    targets = [row[col] for col in self.TARGET_COLUMNS if col in row.index]
                    molecules[mol_name]['targets'] = torch.tensor(targets, dtype=torch.float32)
        
        # Sort conformers by energy (lowest first) and limit count
        for mol_name in molecules:
            conformers = molecules[mol_name]['conformers']
            conformers.sort(key=lambda x: x['conformer_energy'])
            molecules[mol_name]['conformers'] = conformers[:self.max_conformers]
        
        return dict(molecules)
    
    def __len__(self):
        return len(self.mol_names)
    
    def __getitem__(self, idx):
        mol_name = self.mol_names[idx]
        mol_data = self.molecules[mol_name]
        
        return {
            'molecule_id': idx,
            'molecule_name': mol_name,
            'smiles': mol_data['smiles'],
            'conformers': mol_data['conformers'],
            'targets': mol_data['targets']
        }
    
    def get_num_conformers(self, idx):
        """Get number of conformers for a molecule."""
        mol_name = self.mol_names[idx]
        return len(self.molecules[mol_name]['conformers'])
    
    @staticmethod
    def create_splits(
        sdf_path: str,
        csv_path: Optional[str] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42,
        **kwargs
    ):
        """
        Create train/val/test splits of the dataset.
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # First, create a temporary dataset to get molecule count
        temp_supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
        mol_names = set()
        for mol in temp_supplier:
            if mol is not None and mol.HasProp('name'):
                mol_names.add(mol.GetProp('name'))
        
        n_molecules = len(mol_names)
        print(f"Total molecules: {n_molecules}")
        
        # Create random split
        np.random.seed(random_seed)
        indices = np.random.permutation(n_molecules)
        
        n_train = int(n_molecules * train_ratio)
        n_val = int(n_molecules * val_ratio)
        
        train_indices = indices[:n_train].tolist()
        val_indices = indices[n_train:n_train + n_val].tolist()
        test_indices = indices[n_train + n_val:].tolist()
        
        print(f"Split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
        
        # Create datasets
        train_dataset = SDFConformerDataset(
            sdf_path, csv_path, molecule_indices=train_indices, **kwargs
        )
        val_dataset = SDFConformerDataset(
            sdf_path, csv_path, molecule_indices=val_indices, **kwargs
        )
        test_dataset = SDFConformerDataset(
            sdf_path, csv_path, molecule_indices=test_indices, **kwargs
        )
        
        return train_dataset, val_dataset, test_dataset


# Test function
if __name__ == "__main__":
    import os
    
    sdf_path = "datasets/Drugs.sdf"
    csv_path = "datasets/Drugs.csv"
    
    if os.path.exists(sdf_path):
        print("Testing SDFConformerDataset...")
        
        # Test loading a small subset
        dataset = SDFConformerDataset(
            sdf_path=sdf_path,
            csv_path=csv_path,
            max_conformers=5,
            use_sdf_properties=True
        )
        
        print(f"\nDataset size: {len(dataset)}")
        
        # Get first sample
        sample = dataset[0]
        print(f"\nSample 0:")
        print(f"  Molecule name: {sample['molecule_name']}")
        print(f"  SMILES: {sample['smiles'][:50]}...")
        print(f"  Num conformers: {len(sample['conformers'])}")
        print(f"  Targets: {sample['targets']}")
        
        if len(sample['conformers']) > 0:
            conf = sample['conformers'][0]
            print(f"  First conformer:")
            print(f"    Atom features shape: {conf['atom_features'].shape}")
            print(f"    Positions shape: {conf['positions'].shape}")
            print(f"    Edge index shape: {conf['edge_index'].shape}")
    else:
        print(f"SDF file not found: {sdf_path}")
