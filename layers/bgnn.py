import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

class BGNN(nn.Module):
    def __init__(self, node_dim, attr_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        
        # Node transform layers
        self.node_transforms = nn.ModuleList([
            nn.Linear(node_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        
        # Attribute transform layers
        self.attr_transforms = nn.ModuleList([
            nn.Linear(attr_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        
        # Edge attention
        self.edge_attention = nn.Parameter(torch.ones(hidden_dim))
        
    def forward(self, node_feats, attr_feats, edge_index):
        """
        node_feats: [num_nodes, node_dim]
        attr_feats: [num_attrs, attr_dim]
        edge_index: [2, num_edges] - bipartite edges between nodes and attributes
        """
        # Initialize hidden states
        h_nodes = node_feats
        h_attrs = attr_feats
        
        for layer in range(self.num_layers):
            # Node -> Attribute propagation
            node_msg = self.node_transforms[layer](h_nodes)
            node_msg = F.relu(node_msg)
            
            # Compute attention weights
            edge_weights = torch.sum(node_msg[edge_index[0]] * self.edge_attention, dim=1)
            edge_weights = F.softmax(edge_weights, dim=0)
            
            # Update attributes
            attr_msg = scatter_add(
                node_msg[edge_index[0]] * edge_weights.unsqueeze(-1),
                edge_index[1],
                dim=0,
                dim_size=attr_feats.size(0)
            )
            h_attrs = self.attr_transforms[layer](attr_msg)
            h_attrs = F.relu(h_attrs)
            
            # Attribute -> Node propagation
            attr_msg = scatter_add(
                h_attrs[edge_index[1]] * edge_weights.unsqueeze(-1),
                edge_index[0],
                dim=0,
                dim_size=node_feats.size(0)
            )
            h_nodes = h_nodes + attr_msg  # Residual connection
            
        return h_nodes, h_attrs 