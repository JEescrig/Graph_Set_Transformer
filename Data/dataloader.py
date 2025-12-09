import torch
from torch.utilis.data import DAtaset
import pandas as pandas
from typing import List, Dict, Optional
from conformer_generator import conformer_generator
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
            num=conformers=num_conformers,
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
                processed_conformers = self.preprocessor.proces_all_conformers(mol_data)

                self.data.append({
                    'smiles': smles,
                    'mol_data': mol_data,
                    'conformers': preprocessed_conformers,
                    'target': self.targets[idx]
                })

                if (idx + 1) % 100 ==0:
                    print(f"Processed {idx + 1}/{len(self.smiles_list)} molceules")
            except Exception as e:
                print(f"Failed to process molecule {smiles}: {str(e)}")
                continue
        
        print(f"Preprocessing completed. Total molecules: {len(self.data)}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return{
            'molecule_id': idx,
            'smiles': item['smiles'],
            'mol_data': item['mol_data'],
            'conformers': item['conformers'],
            'target':torch.tensor(item['target'], dtype=torch.float32)
        }
    
    
