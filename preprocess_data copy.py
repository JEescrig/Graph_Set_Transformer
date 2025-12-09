"""
Preprocessing script for Buchwald-Hartwig dataset.

1. Loads train_reactions.pkl, valid_reactions.pkl, test_reactions.pkl (with SMILES strings)
2. Converts all SMILES to PyTorch Geometric graphs
3. Saves train_reactions_graphs.pkl, valid_reactions_graphs.pkl, test_reactions_graphs.pkl
"""

import pickle
import torch
import networkx as nx
from torch_geometric.data import Data
from rdkit import Chem, RDLogger
from rdkit.Chem import rdPartialCharges
from tqdm import tqdm
from collections import defaultdict

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
        else:
            edge_attr = None
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def parse_reaction_row(row):
    """Parse reaction DataFrame row to extract SMILES."""
    # reactants and products are already lists of SMILES strings
    reactant_smiles = row['reactants'] if isinstance(row['reactants'], list) else []
    product_smiles = row['products'] if isinstance(row['products'], list) else []
    
    yield_value = row['yield']
    reaction_id = row['reaction_id']
    
    return {
        'reactant_smiles': reactant_smiles,
        'product_smiles': product_smiles,
        'yield': yield_value,
        'reaction_id': reaction_id
    }


def smiles_to_graphs(smiles_list, encoder):
    """Convert a list of SMILES strings to PyG Data objects."""
    graphs = []
    for smiles in smiles_list:
        if not smiles:
            continue
        graph = encoder.smiles_to_graph(smiles)
        if graph is not None:
            graphs.append(graph)
    return graphs


def preprocess_reaction_dataset(input_path, output_path, encoder):
    """
    Load reactions with SMILES, convert to graphs, and save.
    
    Input: PKL file with DataFrame containing SMILES strings
    Output: PKL file with list of reactions containing PyG Data objects
    """
    print(f"\nProcessing: {input_path}")
    
    # Load raw data (pandas DataFrame)
    import pandas as pd
    with open(input_path, 'rb') as f:
        df = pickle.load(f)
    
    print(f"Loaded {len(df)} reactions")
    
    # Process each reaction
    processed_reactions = []
    failed_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting SMILES to graphs"):
        parsed = parse_reaction_row(row)
        
        reactant_graphs = smiles_to_graphs(parsed['reactant_smiles'], encoder)
        product_graphs = smiles_to_graphs(parsed['product_smiles'], encoder)
        
        if len(reactant_graphs) > 0 and len(product_graphs) > 0:
            processed_reactions.append({
                'reaction_id': parsed['reaction_id'],
                'reactants': reactant_graphs,
                'products': product_graphs,
                'yield': parsed['yield'],
            })
        else:
            failed_count += 1
    
    print(f"Successfully processed: {len(processed_reactions)} reactions")
    print(f"Failed conversions: {failed_count} reactions")
    
    # Save processed data as list
    with open(output_path, 'wb') as f:
        pickle.dump(processed_reactions, f)
    
    print(f"Saved to: {output_path}")
    
    return len(processed_reactions)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("Preprocessing Buchwald-Hartwig Dataset")
    print("Converting SMILES strings → PyTorch Geometric graphs")
    print("="*70)
    
    # Initialize encoder
    encoder = SimpleGraphEncoder(charges=True)
    print("\nUsing encoder with Gasteiger charges")
    
    # Define paths
    data_dir = 'Data/buchwald_hartwig_processed'
    
    # Input files (with SMILES)
    input_files = {
        'train': f'{data_dir}/train_reactions.pkl',
        'valid': f'{data_dir}/valid_reactions.pkl',
        'test': f'{data_dir}/test_reactions.pkl',
    }
    
    # Output files (with graphs)
    output_files = {
        'train': f'{data_dir}/train_reactions_graphs.pkl',
        'valid': f'{data_dir}/valid_reactions_graphs.pkl',
        'test': f'{data_dir}/test_reactions_graphs.pkl',
    }
    
    # Process each dataset
    results = {}
    for name in ['train', 'valid', 'test']:
        num_processed = preprocess_reaction_dataset(
            input_files[name],
            output_files[name],
            encoder
        )
        results[name] = num_processed
    
    # Summary
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    print(f"Train set: {results['train']} reactions")
    print(f"Valid set: {results['valid']} reactions")
    print(f"Test set:  {results['test']} reactions")
    print(f"Total:     {sum(results.values())} reactions")
    print("\nNew files created:")
    for name in ['train', 'valid', 'test']:
        print(f"  - {output_files[name]}")
    print("="*70)


if __name__ == "__main__":
    main()

