import math
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv

# Dimension of each attention head in the MHSA
att_HEAD_DIM = 64


class SetGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_gcn_layers=2, mha_dropout=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_gcn_layers = num_gcn_layers
        self.mha_dropout = mha_dropout
       
        self.gcn_layers = nn.ModuleList()

        self.gcn_layers.append(GATConv(in_channels, hidden_channels))

        for _ in range(num_gcn_layers - 2):
            self.gcn_layers.append(GATConv(hidden_channels, hidden_channels))
        
        if num_gcn_layers > 1: 
            self.gcn_layers.append(GATConv(hidden_channels, out_channels))
        else:
            self.gcn_layers.append(GATConv(in_channels, out_channels))
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(
                hidden_channels if i < num_gcn_layers - 1 else out_channels
            )
            for i in range(num_gcn_layers)
        ])

        self.activation = nn.ReLU()

        self.pool = global_mean_pool 

        self.num_heads = max(1, out_channels // att_HEAD_DIM)

        self.mha = nn.MultiheadAttention(
            embed_dim=out_channels,
            num_heads=self.num_heads,
            dropout=self.mha_dropout,
            batch_first=True,
        )

    def forward(self, x, edge_index, batch):
        for i, (gcn_layer, bn) in enumerate(zip(self.gcn_layers, self.batch_norms)):
            x = gcn_layer(x, edge_index)
            x = bn(x)

            if i < self.num_gcn_layers - 1:
                x = self.activation(x)
                x = F.dropout(x, p=0.1, training=self.training)

        graph_embeddings = self.pool(x, batch)

        set_embeddings = graph_embeddings.unsqueeze(0)

        set_embeddings_attn, _ = self.mha(
            set_embeddings, set_embeddings, set_embeddings)

        set_embeddings = set_embeddings + set_embeddings_attn

        graph_embeddings = set_embeddings.squeeze(0)

        graph_embeddings = self.activation(graph_embeddings)

        return graph_embeddings

