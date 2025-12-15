import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import List, Dict, Optional
from conformer_generator import ConformerGenerator
from preprocessing import MoleculePreprocessor

class ConformerDataset(Dataset):
    def __init__(
        self, 
        smiles_list: List[str],
        targets: List[float],
        num_conformers: int = 20,
        max_atomic_num : int = 100,
        add_radius_edges: bool = True,
        radius_cutoff: float = 5.0
    ):
        self.smiles_list = smiles_list
        self.targets = targets
        
        self.generator = ConformerGenerator(
            num_conformers=num_conformers,
            random_seed=42,
            optimize=True
        )

        self.preprocessor = MoleculePreprocessor(
            max_atomic_num=max_atomic_num,
            add_radius_edges=add_radius_edges,
            radius_cutoff=radius_cutoff
        )

        self.data = []
        self._prepare_data()
    
    def _prepare_data(self):
        """Generate conformers and preprocess them"""
        print(f"Preparing {len(self.smiles_list)} molecules...")

        for idx , smiles in enumerate(self.smiles_list):
            try:
                mol_data = self.generator.generate(smiles)
                processed_conformers = self.preprocessor.process_all_conformers(mol_data)

                self.data.append({
                    'smiles': smiles,
                    'mol_data': mol_data,
                    'conformers': processed_conformers,
                    'target': self.targets[idx]
                })

                if (idx + 1) % 100 == 0:
                    print(f"Processed {idx + 1}/{len(self.smiles_list)} molecules")
            except Exception as e:
                print(f"Failed to process molecule {smiles}: {str(e)}")
                continue
        
        print(f"Preprocessing completed. Total molecules: {len(self.data)}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'molecule_id': idx,
            'smiles': item['smiles'],
            'mol_data': item['mol_data'],
            'conformers': item['conformers'],
            'target': torch.tensor(item['target'], dtype=torch.float32)
        }
    
    @classmethod
    def from_csv(cls, csv_path, smiles_col, target_col, **kwargs):
        df = pd.read_csv(csv_path)
        return cls(
            smiles_list=df[smiles_col].tolist(),
            targets=df[target_col].tolist(),
            **kwargs
        )
    def get_num_conformers(self, idx):
        return len(self.data[idx]['conformers'])
    
    def get_conformer(self, mol_idx, conf_idx):
        return self.data[mol_idx]['conformers'][conf_idx]


