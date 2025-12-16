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
    
    def forward(self, x, pos, adj_mat=None, bypass_egnn=False):
        """
        Args:
            x: [num_conformers, num_atoms, input_dim] - 3D batched features
            pos: [num_conformers, num_atoms, 3] - 3D batched positions
            adj_mat: [num_conformers, num_atoms, num_atoms] - adjacency matrix (optional)
            bypass_egnn: if True, skip EGNN layers (for debugging)
            
        Returns:
            embeddings: [num_conformers, hidden_dim] - conformer embeddings
        """
        # Input projection
        h = self.input_proj(x)  # [num_conformers, num_atoms, hidden_dim]
        
        if not bypass_egnn:
            # EGNN layers
            for layer, norm in zip(self.layers, self.norms):
                h, pos = layer(feats=h, coors=pos, adj_mat=adj_mat)
                h = norm(h)
        
        # Pool atoms to conformer embedding (mean over atoms)
        embeddings = h.mean(dim=1)  # [num_conformers, hidden_dim]
        
        return embeddings
