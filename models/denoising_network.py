import torch
import torch.nn as nn
from ..layers.graph_transformer import QueryDrivenGraphTransformer

class DenoisingNetwork(nn.Module):
    def __init__(self, 
                 in_dim, 
                 hidden_dim,
                 num_layers=6,
                 num_heads=8,
                 dropout=0.2):
        super().__init__()
        
        # Graph transformer layers
        self.layers = nn.ModuleList([
            QueryDrivenGraphTransformer(
                in_dim=hidden_dim if i > 0 else in_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for i in range(num_layers)
        ])
        
        # Input projection if needed
        self.input_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
        
        # Output heads for node features and community labels
        self.node_head = nn.Linear(hidden_dim, in_dim)
        self.community_head = nn.Linear(hidden_dim, 2)  # Binary classification
        
    def forward(self, g, query_nodes, t):
        """
        g: Graph object containing node features and edges
        query_nodes: Indices of query nodes
        t: Current timestep
        """
        # Initial projection
        x = self.input_proj(g.v)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, g.e, query_nodes, t)
            
        # Generate predictions
        node_pred = self.node_head(x)
        community_pred = self.community_head(x)
        
        return node_pred, community_pred 