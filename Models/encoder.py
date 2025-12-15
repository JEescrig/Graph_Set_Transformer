import torch
import torch.nn as nn
from .egnn import EGNN
from .deepsets import DeepSets


class ConformerEnsembleEncoder(nn.Module):
    """
    EGNN → (Optional MHA) → DeepSets → Molecule Embeddings
    
    MHA between conformers is optional and disabled by default,
    as the MARCEL paper found DeepSets alone works better.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_egnn_layers: int = 4,
        egnn_dropout: float = 0.0,
        deepsets_dropout: float = 0.0,
        # MHA options (disabled by default)
        use_mha: bool = False,
        mha_num_heads: int = 4,
        mha_num_layers: int = 2,
        mha_dropout: float = 0.1
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.use_mha = use_mha
        
        # EGNN encodes conformers
        self.egnn = EGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_egnn_layers,
            dropout=egnn_dropout,
            pool='mean'
        )
        
        # Project to output dim if needed
        if hidden_dim != output_dim:
            self.proj = nn.Linear(hidden_dim, output_dim)
        else:
            self.proj = nn.Identity()
        
        # Optional MHA between conformers
        if use_mha:
            self.mha = ConformerMHA(
                hidden_dim=output_dim,
                num_heads=mha_num_heads,
                num_layers=mha_num_layers,
                dropout=mha_dropout
            )
        
        # DeepSets aggregates conformers to molecules
        self.deepsets = DeepSets(
            hidden_dim=output_dim,
            dropout=deepsets_dropout
        )
    
    def forward(self, x, pos, edge_index, batch, conformer_to_molecule):
        """
        Args:
            x: [num_nodes, input_dim]
            pos: [num_nodes, 3]
            edge_index: [2, num_edges]
            batch: [num_nodes] - conformer index for each node
            conformer_to_molecule: [num_conformers] - molecule index for each conformer
            
        Returns:
            molecule_embeddings: [num_molecules, output_dim]
        """
        # EGNN: nodes → conformer embeddings
        conformer_embeddings = self.egnn(x, pos, edge_index, batch)
        
        # Project
        conformer_embeddings = self.proj(conformer_embeddings)
        
        # Optional MHA: conformers interact within each molecule
        if self.use_mha:
            conformer_embeddings = self.mha(
                conformer_embeddings,
                conformer_to_molecule
            )
        
        # DeepSets: conformers → molecule embeddings
        molecule_embeddings = self.deepsets(conformer_embeddings, conformer_to_molecule)
        
        return molecule_embeddings


class ConformerMHA(nn.Module):
    """
    Multi-head attention between conformers of the SAME molecule.
    
    Conformers only attend to other conformers of their molecule,
    not to conformers of other molecules.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.layers.append(
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
    
    def forward(self, conformer_embeddings, conformer_to_molecule):
        """
        Apply MHA between conformers of the same molecule.
        
        Args:
            conformer_embeddings: [num_conformers, hidden_dim]
            conformer_to_molecule: [num_conformers] - molecule index for each conformer
            
        Returns:
            updated_embeddings: [num_conformers, hidden_dim]
        """
        device = conformer_embeddings.device
        num_molecules = conformer_to_molecule.max().item() + 1
        
        # Count conformers per molecule
        counts = torch.bincount(conformer_to_molecule, minlength=num_molecules)
        max_conformers = counts.max().item()
        
        # Create padded tensor and mask
        padded = torch.zeros(
            num_molecules, max_conformers, self.hidden_dim,
            device=device
        )
        mask = torch.ones(
            num_molecules, max_conformers,
            dtype=torch.bool,
            device=device
        )
        
        # Track positions for filling and unpacking
        positions = torch.zeros(num_molecules, dtype=torch.long, device=device)
        conformer_indices = []
        
        # Fill padded tensor
        for i, (emb, mol_idx) in enumerate(zip(conformer_embeddings, conformer_to_molecule)):
            pos = positions[mol_idx]
            padded[mol_idx, pos] = emb
            mask[mol_idx, pos] = False
            conformer_indices.append((mol_idx.item(), pos.item()))
            positions[mol_idx] += 1
        
        # Apply MHA layers
        x = padded
        for layer, norm in zip(self.layers, self.norms):
            attn_out, _ = layer(x, x, x, key_padding_mask=mask)
            x = x + attn_out
            x = norm(x)
        
        # Unpack back to original order
        updated_embeddings = torch.zeros_like(conformer_embeddings)
        for i, (mol_idx, pos) in enumerate(conformer_indices):
            updated_embeddings[i] = x[mol_idx, pos]
        
        return updated_embeddings


class MoleculeInteractionNetwork(nn.Module):
    """
    Multi-head attention for molecule interactions (Stage 2).
    
    Molecules within the same set interact with each other.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.layers.append(
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
    
    def forward(self, x, molecule_to_set=None):
        """
        Args:
            x: Either:
               - [batch_size, num_molecules, hidden_dim] (already grouped)
               - [num_molecules, hidden_dim] (flat, needs molecule_to_set)
            molecule_to_set: [num_molecules] - which set each molecule belongs to
                            (only needed if x is flat)
        
        Returns:
            Same shape as input
        """
        # If flat input, group into sets
        if x.dim() == 2 and molecule_to_set is not None:
            x, mask, indices = self._group_into_sets(x, molecule_to_set)
            was_flat = True
        else:
            mask = None
            was_flat = False
        
        # Apply MHA layers
        for layer, norm in zip(self.layers, self.norms):
            attn_out, _ = layer(x, x, x, key_padding_mask=mask)
            x = x + attn_out
            x = norm(x)
        
        # If input was flat, ungroup back
        if was_flat:
            x = self._ungroup_from_sets(x, indices)
        
        return x
    
    def _group_into_sets(self, molecule_embeddings, molecule_to_set):
        """Group flat molecule embeddings into sets with padding."""
        device = molecule_embeddings.device
        num_sets = molecule_to_set.max().item() + 1
        
        # Count molecules per set
        counts = torch.bincount(molecule_to_set, minlength=num_sets)
        max_molecules = counts.max().item()
        
        # Create padded tensor and mask
        padded = torch.zeros(
            num_sets, max_molecules, self.hidden_dim,
            device=device
        )
        mask = torch.ones(
            num_sets, max_molecules,
            dtype=torch.bool,
            device=device
        )
        
        # Track positions
        positions = torch.zeros(num_sets, dtype=torch.long, device=device)
        indices = []
        
        # Fill padded tensor
        for i, (emb, set_idx) in enumerate(zip(molecule_embeddings, molecule_to_set)):
            pos = positions[set_idx]
            padded[set_idx, pos] = emb
            mask[set_idx, pos] = False
            indices.append((set_idx.item(), pos.item()))
            positions[set_idx] += 1
        
        return padded, mask, indices
    
    def _ungroup_from_sets(self, padded, indices):
        """Ungroup padded sets back to flat molecule embeddings."""
        num_molecules = len(indices)
        device = padded.device
        
        molecule_embeddings = torch.zeros(
            num_molecules, self.hidden_dim,
            device=device
        )
        
        for i, (set_idx, pos) in enumerate(indices):
            molecule_embeddings[i] = padded[set_idx, pos]
        
        return molecule_embeddings