"""
EGNN Wrapper for egnn_pytorch
=============================
Processes batched conformers from the SAME molecule (all same num_atoms).
Input: 3D tensor [num_conformers, num_atoms, features]
"""

import torch
import torch.nn as nn
from egnn_pytorch import EGNN as EGNNLayer


class EGNN(nn.Module):
    """
    EGNN wrapper for processing conformer batches from a single molecule.
    
    Expects 3D input: [num_conformers, num_atoms, features]
    All conformers must have the same number of atoms (same molecule).
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 4,
        m_dim: int = 32,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # EGNN layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.layers.append(
                EGNNLayer(
                    dim=hidden_dim,
                    edge_dim=0,
                    m_dim=m_dim,
                    dropout=dropout,
                    norm_coors=True,
                    update_coors=False
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
    
    def forward(self, x, pos, adj_mat=None):
        """
        Args:
            x: [num_conformers, num_atoms, input_dim] - 3D batched features
            pos: [num_conformers, num_atoms, 3] - 3D batched positions
            adj_mat: [num_conformers, num_atoms, num_atoms] - adjacency matrix (optional)
            
        Returns:
            embeddings: [num_conformers, hidden_dim] - conformer embeddings
        """
        # Input projection
        h = self.input_proj(x)  # [num_conformers, num_atoms, hidden_dim]
        
        # EGNN layers
        for layer, norm in zip(self.layers, self.norms):
            # egnn_pytorch expects: feats [B, N, D], coors [B, N, 3]
            h_update, pos = layer(feats=h, coors=pos, adj_mat=adj_mat)
            h = h + h_update  # Residual
            h = norm(h)
        
        # Pool atoms to conformer embedding (mean over atoms)
        embeddings = h.mean(dim=1)  # [num_conformers, hidden_dim]
        
        return embeddings


class ConformerBatchEGNN(nn.Module):
    """
    Processes one molecule's conformers at a time.
    
    For use when batching conformers of the SAME molecule together.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 4,
        m_dim: int = 32,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.egnn = EGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            m_dim=m_dim,
            dropout=dropout
        )
        self.hidden_dim = hidden_dim
    
    def forward(self, conformers_list):
        """
        Process a list of conformer batches (one batch per molecule).
        
        Args:
            conformers_list: List of dicts, each containing:
                - 'x': [num_conformers, num_atoms, input_dim]
                - 'pos': [num_conformers, num_atoms, 3]
                - 'adj_mat': [num_conformers, num_atoms, num_atoms] (optional)
                
        Returns:
            all_embeddings: [total_conformers, hidden_dim]
            conformer_to_molecule: [total_conformers] - which molecule each conformer belongs to
        """
        all_embeddings = []
        conformer_to_molecule = []
        
        for mol_idx, conf_batch in enumerate(conformers_list):
            x = conf_batch['x']
            pos = conf_batch['pos']
            adj_mat = conf_batch.get('adj_mat', None)
            
            # Process this molecule's conformers
            embeddings = self.egnn(x, pos, adj_mat)  # [num_conf, hidden_dim]
            
            all_embeddings.append(embeddings)
            conformer_to_molecule.extend([mol_idx] * embeddings.size(0))
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        conformer_to_molecule = torch.tensor(conformer_to_molecule, device=all_embeddings.device)
        
        return all_embeddings, conformer_to_molecule
