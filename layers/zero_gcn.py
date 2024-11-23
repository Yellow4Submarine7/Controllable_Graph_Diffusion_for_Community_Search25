import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class ZeroGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        
        # Initialize GCN layers with zero weights
        self.convs = nn.ModuleList([
            GCNConv(
                in_dim if i == 0 else hidden_dim,
                hidden_dim,
                bias=False
            ) for i in range(num_layers)
        ])
        
        # Initialize weights to zero
        self.reset_parameters()
        
    def reset_parameters(self):
        for conv in self.convs:
            nn.init.zeros_(conv.weight)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
                
    def forward(self, x, edge_index):
        """
        x: Node features [num_nodes, in_dim]
        edge_index: Graph connectivity [2, num_edges]
        """
        h = x
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index)
            if i < self.num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=0.5, training=self.training)
        return h 