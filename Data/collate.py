# Collate function for conformer batches
# Groups conformers by molecule into 3D tensors for egnn_pytorch

import torch
from torch_geometric.data import Data, Batch


def collate_molecule_batch(batch, max_conformers=None):
    """
    Collate function that groups conformers by molecule.
    
    Each molecule's conformers are stacked into 3D tensors.
    
    Args:
        batch: List of samples from SDFConformerDataset
        max_conformers: Optional limit on conformers per molecule
        
    Returns:
        Dict with:
        - conformer_batches: List of dicts, one per molecule
        - targets: [batch_size, num_targets]
        - molecule_names: List of molecule names
    """
    conformer_batches = []
    targets = []
    molecule_names = []
    molecule_ids = []
    
    for sample in batch:
        molecule_names.append(sample['molecule_name'])
        molecule_ids.append(sample['molecule_id'])
        targets.append(sample['targets'])
        
        conformers = sample['conformers']
        if max_conformers is not None and len(conformers) > max_conformers:
            conformers = conformers[:max_conformers]
        
        if len(conformers) == 0:
            continue
        
        # Stack conformers into 3D tensors
        # All conformers of same molecule have same num_atoms
        x_list = [c['atom_features'] for c in conformers]
        pos_list = [c['positions'] for c in conformers]
        
        x_stacked = torch.stack(x_list, dim=0)     # [num_conf, num_atoms, feat_dim]
        pos_stacked = torch.stack(pos_list, dim=0) # [num_conf, num_atoms, 3]
        
        # Create adjacency matrix from edge_index (same for all conformers)
        num_atoms = x_stacked.size(1)
        edge_index = conformers[0]['edge_index']
        adj_mat = torch.zeros(num_atoms, num_atoms)
        if edge_index.size(1) > 0:
            adj_mat[edge_index[0], edge_index[1]] = 1.0
        adj_mat = adj_mat.unsqueeze(0).expand(len(conformers), -1, -1)  # [num_conf, num_atoms, num_atoms]
        
        conformer_batches.append({
            'x': x_stacked,
            'pos': pos_stacked,
            'adj_mat': adj_mat,
            'num_conformers': len(conformers)
        })
    
    targets = torch.stack(targets, dim=0)
    
    return {
        'conformer_batches': conformer_batches,
        'targets': targets,
        'molecule_names': molecule_names,
        'molecule_ids': molecule_ids,
        'batch_size': len(batch)
    }


def collate_conformer_batch(batch, max_conformers=None):
    """
    Original collate function - flattens all conformers into PyG batch.
    
    Use this with custom EGNN that works with 2D tensors.
    """
    all_graphs = []
    molecule_ids = []
    molecule_names = []
    targets = []
    conformer_to_molecule = []

    for mol_idx, sample in enumerate(batch):
        molecule_ids.append(sample['molecule_id'])
        molecule_names.append(sample['molecule_name'])
        targets.append(sample['targets'])

        conformers = sample['conformers']
        if max_conformers is not None and len(conformers) > max_conformers:
            conformers = conformers[:max_conformers]
        
        for conf in conformers:
            # Create PyG data object
            graph = Data(
                x=conf['atom_features'],
                pos=conf['positions'],
                edge_index=conf['edge_index']
            )
            all_graphs.append(graph)
            conformer_to_molecule.append(mol_idx)

    batched_graph = Batch.from_data_list(all_graphs)
    targets = torch.stack(targets)

    return {
        'graph': batched_graph,
        'molecule_ids': molecule_ids,
        'molecule_names': molecule_names,
        'targets': targets,
        'conformer_to_molecule': torch.tensor(conformer_to_molecule, dtype=torch.long),
        'batch_size': len(batch),
        'num_conformers': len(all_graphs)
    }
