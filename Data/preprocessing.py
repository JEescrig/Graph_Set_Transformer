import torch
import numpy as np
import rdkit.Chem.AllChem as AllChem
import torch.nn.functional as F

class MoleculePreprocessor:
    def __init__(self, max_atomic_num=100, add_radius_edges=True, radius_cutoff=5.0):
        """Initialize the molecule preprocessor"""
        self.add_radius_edges = add_radius_edges
        self.radius_cutoff = radius_cutoff
        self.max_atomic_num = max_atomic_num

    def get_atom_features(self, mol):
        features = []
        for atom in mol.GetAtoms():
            atom_features = []

            # Atomic number
            atomic_num = atom.GetAtomicNum()
            atom_features.append(atomic_num)

            #Degree
            atom_features.append(atom.GetDegree())

            # Formal charge
            atom_features.append(atom.GetFormalCharge())

            #Number of Hydrogens
            atom_features.append(atom.GetTotalNumHs())

            #Is aromatic
            atom_features.append(1 if atom.GetIsAromatic() else 0)

            #Is in ring
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
        conformer = mol_data['conformers'][conformer_idx]
        positions = torch.tensor(conformer['coords'], dtype=torch.float)

        atom_features = self.get_atom_features(mol)
        edge_index = self.get_edge_index(mol_data['bonds'], positions)

        energy = conformer['energy']

        return{
            'atom_features': atom_features,
            'edge_index': edge_index,
            'positions': positions,
            'energy': energy
        }
    
    def process_all_conformers(self, mol_data, mol=None):
        processed = []

        for i in range(len(mol_data['conformers'])):
            processed.append(self.process_conformer(mol_data, conformer_idx=i, mol=mol))
        
        return processed