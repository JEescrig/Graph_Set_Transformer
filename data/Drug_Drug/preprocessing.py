"""
Preprocessing script for Drug-Drug Interaction dataset.

This script:
1. Loads train/valid/test_reactions.pkl (DataFrames with Drug1, Drug2 SMILES, and Y label)
2. Converts all SMILES to PyTorch Geometric graphs
3. Saves train/valid/test_reactions_graphs.pkl with graph representations
"""

import pickle
import torch
import networkx as nx
from torch_geometric.data import Data
from rdkit import Chem, RDLogger
from rdkit.Chem import rdPartialCharges
from tqdm import tqdm
from collections import defaultdict
import pandas as pd

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')


# ============================================================================
# ATOM AND BOND FEATURE EXTRACTION
# ============================================================================

def get_atomic_invariants_as_dict(atom, charges=False):
    """Extract atom features as a dictionary."""
    features = {
        'atomic_num': atom.GetAtomicNum(),
        'degree': atom.GetDegree(),
        'formal_charge': atom.GetFormalCharge(),
        'hybridization': int(atom.GetHybridization()),
        'is_aromatic': int(atom.GetIsAromatic()),
        'num_hs': atom.GetTotalNumHs(),
        'num_radical_electrons': atom.GetNumRadicalElectrons(),
        'is_in_ring': int(atom.IsInRing()),
    }
    
    if charges:
        try:
            charge = float(atom.GetProp('_GasteigerCharge'))
            # Replace NaN/inf with 0
            if not (charge == charge) or abs(charge) == float('inf'):
                charge = 0.0
            features['gasteiger_charge'] = charge
        except:
            features['gasteiger_charge'] = 0.0
    
    return features


def get_bond_invariants_as_dict(bond):
    """Extract bond features as a dictionary."""
    features = {
        'bond_type': int(bond.GetBondType()),
        'is_conjugated': int(bond.GetIsConjugated()),
        'is_in_ring': int(bond.IsInRing()),
        'stereo': int(bond.GetStereo()),
    }
    
    return features


# ============================================================================
# GRAPH ENCODER CLASS
# ============================================================================

class SimpleGraphEncoder:
    """Converts SMILES strings to PyTorch Geometric Data objects."""
    
    def __init__(self, charges=False):
        self.charges = charges
    
    def smiles_to_graph(self, smiles):
        """Convert a SMILES string to a PyTorch Geometric Data object."""
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            mol = Chem.MolFromSmiles(smiles.replace("[NH+2]", "[NH+1]"))
            if mol is None:
                return None
        
        if self.charges:
            try:
                rdPartialCharges.ComputeGasteigerCharges(mol)
            except:
                pass
        
        # Convert to NetworkX graph
        G = nx.Graph()
        
        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(), **get_atomic_invariants_as_dict(atom, self.charges))
        
        for bond in mol.GetBonds():
            G.add_edge(
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                **get_bond_invariants_as_dict(bond)
            )
        
        return self.nx_to_pyg(G)
    
    def nx_to_pyg(self, G):
        """Convert NetworkX graph to PyTorch Geometric Data object."""
        if G.number_of_nodes() == 0:
            return None
        
        G = G.to_directed() if not nx.is_directed(G) else G
        
        # Create edge index
        mapping = dict(zip(G.nodes(), range(G.number_of_nodes())))
        edge_index = torch.empty((2, G.number_of_edges()), dtype=torch.long)
        for i, (src, dst) in enumerate(G.edges()):
            edge_index[0, i] = mapping[src]
            edge_index[1, i] = mapping[dst]
        
        # Extract node features
        data_dict = defaultdict(list)
        node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
        
        for _, feat_dict in G.nodes(data=True):
            for key, value in feat_dict.items():
                data_dict[str(key)].append(value)
        
        # Convert to tensors
        node_features = []
        for key in node_attrs:
            values = data_dict[key]
            if isinstance(values[0], bool):
                values = [int(v) for v in values]
            tensor = torch.tensor(values, dtype=torch.float).view(-1, 1)
            node_features.append(tensor)
        
        x = torch.cat(node_features, dim=-1)
        
        # Extract edge features
        edge_data_dict = defaultdict(list)
        
        if G.number_of_edges() > 0:
            edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())
            
            for _, _, feat_dict in G.edges(data=True):
                for key, value in feat_dict.items():
                    edge_data_dict[str(key)].append(value)
            
            edge_features = []
            for key in edge_attrs:
                values = edge_data_dict[key]
                if isinstance(values[0], bool):
                    values = [int(v) for v in values]
                tensor = torch.tensor(values, dtype=torch.float).view(-1, 1)
                edge_features.append(tensor)
            
            edge_attr = torch.cat(edge_features, dim=-1)
            num_edge_features = edge_attr.shape[1]
        else:
            edge_attr = None
            num_edge_features = 4  # Default: bond_type, conjugated, in_ring, stereo
        
        if edge_attr is not None:
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        else:
            # Empty tensor for graphs with no edges
            empty_edge_attr = torch.empty((0, num_edge_features), dtype=torch.float)
            return Data(x=x, edge_index=edge_index, edge_attr=empty_edge_attr)  


# ============================================================================
# PREPROCESSING FUNCTIONS FOR DRUG-DRUG INTERACTIONS
# ============================================================================

def parse_drug_interaction_row(row):
    """Parse drug interaction DataFrame row to extract SMILES and label."""
    drug1_smiles = row['Drug1']        # SMILES for drug 1
    drug2_smiles = row['Drug2']        # SMILES for drug 2
    interaction_label = row['Y']       # Interaction label/yield
    
    # Get IDs if they exist
    drug1_id = row.get('Drug1_ID', f'drug1_{row.name}')
    drug2_id = row.get('Drug2_ID', f'drug2_{row.name}')
    
    return {
        'drug1_smiles': drug1_smiles,
        'drug2_smiles': drug2_smiles,
        'drug1_id': drug1_id,
        'drug2_id': drug2_id,
        'yield': interaction_label,  # Keep name consistent with training code
        'interaction_id': f'{drug1_id}_{drug2_id}'
    }


def preprocess_drug_interaction_dataset(input_path, output_path, encoder):
    """
    Load drug-drug interactions with SMILES, convert to graphs, and save.
    
    Input: PKL file with DataFrame containing Drug1, Drug2 SMILES, and Y label
    Output: PKL file with list of interactions containing PyG Data objects
    """
    print(f"\nProcessing: {input_path}")
    
    # Load raw data (pandas DataFrame)
    with open(input_path, 'rb') as f:
        df = pickle.load(f)
    
    print(f"Loaded {len(df)} drug-drug interactions")
    
    # Process each interaction
    processed_interactions = []
    failed_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting SMILES to graphs"):
        parsed = parse_drug_interaction_row(row)
        
        # Convert both drugs to graphs
        drug1_graph = encoder.smiles_to_graph(parsed['drug1_smiles'])
        drug2_graph = encoder.smiles_to_graph(parsed['drug2_smiles'])
        
        if drug1_graph is not None and drug2_graph is not None:
            processed_interactions.append({
                'reaction_id': parsed['interaction_id'],  # Keep consistent naming
                'reactants': [drug1_graph],               # Drug 1 as "reactant"
                'products': [drug2_graph],                # Drug 2 as "product"
                'yield': parsed['yield'],                 # Interaction label
                'drug1_id': parsed['drug1_id'],
                'drug2_id': parsed['drug2_id'],
            })
        else:
            failed_count += 1
            if failed_count <= 5:  # Show first few failures
                print(f"  Failed to convert: {parsed['drug1_id']} + {parsed['drug2_id']}")
    
    print(f"Successfully processed: {len(processed_interactions)} interactions")
    print(f"Failed conversions: {failed_count} interactions")
    
    # Save processed data as list
    with open(output_path, 'wb') as f:
        pickle.dump(processed_interactions, f)
    
    print(f"Saved to: {output_path}")
    
    return len(processed_interactions)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("Preprocessing Drug-Drug Interaction Dataset")
    print("Converting SMILES strings → PyTorch Geometric graphs")
    print("="*70)
    
    # Initialize encoder
    encoder = SimpleGraphEncoder(charges=True)
    print("\nUsing encoder with Gasteiger charges")
    

    
    # Input files (with SMILES)
    input_files = {
        'train': f'train_reactions.pkl',
        'valid': f'valid_reactions.pkl',
        'test': f'test_reactions.pkl',
    }
    
    # Output files (with graphs)
    output_files = {
        'train': f'train_reactions_graphs.pkl',
        'valid': f'valid_reactions_graphs.pkl',
        'test': f'test_reactions_graphs.pkl',
    }
    
    # Process each dataset
    results = {}
    for name in ['train', 'valid', 'test']:
        num_processed = preprocess_drug_interaction_dataset(
            input_files[name],
            output_files[name],
            encoder
        )
        results[name] = num_processed
    
    # Summary
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    print(f"Train set: {results['train']} interactions")
    print(f"Valid set: {results['valid']} interactions")
    print(f"Test set:  {results['test']} interactions")
    print(f"Total:     {sum(results.values())} interactions")
    print("\nNew files created:")
    for name in ['train', 'valid', 'test']:
        print(f"  - {output_files[name]}")
    print("="*70)


if __name__ == "__main__":
    main()
