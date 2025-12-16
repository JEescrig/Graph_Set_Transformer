import torch
import numpy as np
import rdkit.Chem.AllChem as AllChem
from rdkit import Chem
import torch.nn.functional as F

class MoleculePreprocessor:
    def __init__(self, max_atomic_num=100, add_radius_edges=True, radius_cutoff=5.0):
        """Initialize the molecule preprocessor"""
        self.add_radius_edges = add_radius_edges
        self.radius_cutoff = radius_cutoff
        self.max_atomic_num = max_atomic_num

    # Hybridization encoding (one-hot would be better, but using ordinal for simplicity)
    HYBRIDIZATION_MAP = {
        Chem.rdchem.HybridizationType.SP: 1,
        Chem.rdchem.HybridizationType.SP2: 2,
        Chem.rdchem.HybridizationType.SP3: 3,
        Chem.rdchem.HybridizationType.SP3D: 4,
        Chem.rdchem.HybridizationType.SP3D2: 5,
    }

    def get_atom_features(self, mol):
        """
        Get atom features matching MARCEL paper (Table S1).
        Returns 9 features per atom:
        1. AtomicNum - Atomic number
        2. ChiralTag - Indicator of chirality
        3. TotalDegree - Sum of implicit and explicit bonds
        4. FormalCharge - Formal charge
        5. TotalNumHs - Total hydrogen count
        6. NumRadicalElectrons - Unpaired electrons
        7. Hybridization - Orbital hybridization type
        8. IsAromatic - In aromatic ring
        9. IsInRing - In any ring
        """
        features = []
        for atom in mol.GetAtoms():
            atom_features = []

            # 1. Atomic number
            atom_features.append(atom.GetAtomicNum())

            # 2. Chiral tag (0=unspecified, 1=CCW, 2=CW, 3=other)
            atom_features.append(int(atom.GetChiralTag()))

            # 3. Total degree (implicit + explicit bonds)
            atom_features.append(atom.GetTotalDegree())

            # 4. Formal charge
            atom_features.append(atom.GetFormalCharge())

            # 5. Total number of hydrogens
            atom_features.append(atom.GetTotalNumHs())

            # 6. Number of radical electrons
            atom_features.append(atom.GetNumRadicalElectrons())

            # 7. Hybridization (ordinal encoded)
            hyb = atom.GetHybridization()
            atom_features.append(self.HYBRIDIZATION_MAP.get(hyb, 0))

            # 8. Is aromatic
            atom_features.append(1 if atom.GetIsAromatic() else 0)

            # 9. Is in ring
            atom_features.append(1 if atom.IsInRing() else 0)

            features.append(atom_features)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def get_edge_index_from_bonds(self, bonds):
        if len(bonds) == 0:
            return torch.zeros((2, 0), dtype=torch.long)
        
        sources = []
        targets = []

        for (i, j) in bonds:
            sources.extend([i, j])
            targets.extend([j, i])
        
        return torch.tensor([sources, targets], dtype=torch.long)
    
    def get_radius_edges(self, positions, cutoff):
        num_atoms = positions.shape[0]
        sources = []
        targets = []

        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                
                diff = positions[i] - positions[j]
                distance = torch.sqrt(torch.sum(diff ** 2))

                if distance < cutoff:

                    sources.extend([i, j])
                    targets.extend([j, i])
        
        if len(sources) == 0:
            return torch.zeros((2, 0), dtype=torch.long)
        
        return torch.tensor([sources, targets], dtype=torch.long)
    
    def get_edge_index(self, bonds, positions=None):

        edge_index = self.get_edge_index_from_bonds(bonds)

        if self.add_radius_edges and positions is not None:
            radius_edges = self.get_radius_edges(positions, self.radius_cutoff)

            edge_index = torch.cat([edge_index, radius_edges], dim=1)

            edge_index = torch.unique(edge_index, dim=1)
        
        return edge_index
    
    def process_conformer(self, mol_data, conformer_idx=0, mol=None):
        if mol is None:
            mol = Chem.MolFromSmiles(mol_data['smiles'])
            mol = Chem.AddHs(mol)

        conformer = mol_data['conformers'][conformer_idx]
        positions = torch.tensor(conformer['coords'], dtype=torch.float)

        atom_features = self.get_atom_features(mol)
        edge_index = self.get_edge_index(mol_data['bonds'], positions)

        energy = conformer['energy']

        return {
            'atom_features': atom_features,
            'edge_index': edge_index,
            'positions': positions,
            'energy': energy
        }
    
    def process_all_conformers(self, mol_data, mol=None):
        processed = []
        
        if mol is None:
            mol = Chem.MolFromSmiles(mol_data['smiles'])
            mol = Chem.AddHs(mol)

        for i in range(len(mol_data['conformers'])):
            processed.append(self.process_conformer(mol_data, conformer_idx=i, mol=mol))
        
        return processed