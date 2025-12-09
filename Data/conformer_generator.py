from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np 

class ConformerGenerator:
    def __init__(self, num_conformers=20, random_seed=42, optimize=True):
        self.num_conformers = num_conformers
        self.random_seed = random_seed
        self.optimize = optimize
    
    def generate(self, smiles):
        """Conformer generation from smiles"""

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid smiles: ({smiles})")
        
        mol = Chem.AddHs(mol)

        params = AllChem.ETKDGv3()

        params.randomSeed = self.random_seed
        params.pruneRmsThresh = 0.5
        
        conformer_ids = AllChem.EmbedMultipleConfs(
            mol,
            numConfs=self.num_conformers,
            params=params
        )
        if len(conformer_ids) == 0:
            raise ValueError(f"Failed to generate conformers for smiles: ({smiles})")
        
        conformer_data = []
        for conformer_id in conformer_ids:
            if self.optimize:
                ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol), confId=conformer_id)

                if ff is None:
                    ff = AllChem.UFFGetMoleculeForceField(mol, confId=conformer_id)
                
                if ff is not None:
                    ff.Minimize()
                    energy = ff.CalcEnergy()
                else:
                    energy = float('inf')
            
            conf = mol.GetConformer(conformer_id)
            coords = conf.GetPositions()
            conformer_data.append({
                'coords': coords,
                'energy': energy
            })
        
        atom_types = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

        bonds = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bonds.append((i, j))
        
        conformer_data.sort(key=lambda x: x['energy'])
        
        return {
            'smiles': smiles,
            'atom_types': atom_types,
            'bonds': bonds,
            'conformers': conformer_data
        }