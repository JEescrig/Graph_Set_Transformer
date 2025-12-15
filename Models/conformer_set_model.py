"""
Conformer Set Model for Multi-Target Molecular Property Prediction
===================================================================
Uses EGNN → MHA → DeepSets → Prediction Head architecture.

Works with egnn_pytorch by processing conformers grouped by molecule.
"""

import torch
import torch.nn as nn

from .egnn import EGNN
from .deepsets import DeepSets
from .encoder import ConformerMHA


class ConformerSetModel(nn.Module):
    """
    Full model for predicting molecular properties from conformer ensembles.
    
    Architecture:
        1. EGNN: Encodes conformers (per molecule batch) → conformer embeddings
        2. ConformerMHA: Conformers attend to each other within molecule
        3. DeepSets: Aggregates conformer embeddings → molecule embedding
        4. Prediction Head: Maps molecule embedding → target properties
    """
    
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_targets: int = 4,
        num_egnn_layers: int = 4,
        egnn_dropout: float = 0.1,
        use_conformer_mha: bool = True,
        num_mha_heads: int = 4,
        num_mha_layers: int = 1,
        mha_dropout: float = 0.1,
        deepsets_dropout: float = 0.1,
        head_dropout: float = 0.2
    ):
        super().__init__()
        
        self.num_targets = num_targets
        self.use_conformer_mha = use_conformer_mha
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # EGNN encodes conformer graphs
        self.egnn = EGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_egnn_layers,
            dropout=egnn_dropout
        )
        
        # Project to output dim if needed
        if hidden_dim != output_dim:
            self.proj = nn.Linear(hidden_dim, output_dim)
        else:
            self.proj = nn.Identity()
        
        # Conformer MHA - conformers attend to each other
        if use_conformer_mha:
            self.conformer_mha = ConformerMHA(
                hidden_dim=output_dim,
                num_heads=num_mha_heads,
                num_layers=num_mha_layers,
                dropout=mha_dropout
            )
        
        # DeepSets aggregates conformers to molecules
        self.deepsets = DeepSets(
            hidden_dim=output_dim,
            dropout=deepsets_dropout
        )
        
        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim // 2, num_targets)
        )
    
    def forward(self, batch):
        """
        Args:
            batch: Dict from collate_molecule_batch containing:
                - conformer_batches: List of dicts with 'x', 'pos', 'adj_mat' per molecule
                - targets: [batch_size, num_targets]
                
        Returns:
            predictions: [batch_size, num_targets]
        """
        conformer_batches = batch['conformer_batches']
        device = batch['targets'].device
        
        all_conformer_embeddings = []
        conformer_to_molecule = []
        
        # Process each molecule's conformers
        for mol_idx, conf_batch in enumerate(conformer_batches):
            x = conf_batch['x'].to(device)
            pos = conf_batch['pos'].to(device)
            adj_mat = conf_batch['adj_mat'].to(device) if 'adj_mat' in conf_batch else None
            
            # EGNN: [num_conf, num_atoms, feat] → [num_conf, hidden_dim]
            embeddings = self.egnn(x, pos, adj_mat)
            
            all_conformer_embeddings.append(embeddings)
            conformer_to_molecule.extend([mol_idx] * embeddings.size(0))
        
        # Concatenate all conformer embeddings
        conformer_embeddings = torch.cat(all_conformer_embeddings, dim=0)
        conformer_to_molecule = torch.tensor(conformer_to_molecule, device=device, dtype=torch.long)
        
        # Project
        conformer_embeddings = self.proj(conformer_embeddings)
        
        # Conformer MHA
        if self.use_conformer_mha:
            conformer_embeddings = self.conformer_mha(
                conformer_embeddings, conformer_to_molecule
            )
        
        # DeepSets: conformers → molecule embeddings
        molecule_embeddings = self.deepsets(conformer_embeddings, conformer_to_molecule)
        
        # Predict targets
        predictions = self.prediction_head(molecule_embeddings)
        
        return predictions