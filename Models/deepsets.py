import torch
import torch.nn as nn
from torch_geometric.utils import scatter


class DeepSets(nn.Module):
    """DeepSets aggregator for conformer embeddings."""
    
    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.0,
        use_layer_norm: bool = True
    ):
       
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm
        
        # h(): transforms conformer embedding
        self.h = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
     
    
        self.norm = nn.LayerNorm(hidden_dim)
        
        # g(): transforms aggregated embedding
        self.g = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, conformer_embeddings, conformer_to_molecule):
        """Aggregate conformer embeddings into molecule embeddings."""
        # Transform each conformer
        transformed = self.h(conformer_embeddings)
        
        summed = scatter(
            transformed,
            conformer_to_molecule,
            dim=0,
            reduce='sum'
        )
        
        # Normalize if enabled
        summed = self.norm(summed)
        
        # Final transformation
        output = self.g(summed)
        
        return output