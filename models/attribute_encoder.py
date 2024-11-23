import torch
import torch.nn as nn
from ..layers.bgnn import BGNN

class AttributeEncoder(nn.Module):
    def __init__(self, node_dim, attr_dim, hidden_dim, num_layers=2):
        super().__init__()
        
        # BGNN for attribute-node alignment
        self.bgnn = BGNN(
            node_dim=node_dim,
            attr_dim=attr_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        
        # Attribute embedding
        self.attr_embedding = nn.Embedding(attr_dim, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, node_dim)
        
    def forward(self, g, query_attrs):
        """
        g: Graph object
        query_attrs: Query attribute indices
        """
        # Get attribute embeddings
        attr_feats = self.attr_embedding(query_attrs)
        
        # Build bipartite edges between nodes and attributes
        edge_index = self._build_bipartite_edges(g.v.size(0), query_attrs.size(0))
        
        # Apply BGNN
        node_feats, _ = self.bgnn(g.v, attr_feats, edge_index)
        
        # Project back to node feature space
        enhanced_feats = self.output_proj(node_feats)
        
        return enhanced_feats
        
    def _build_bipartite_edges(self, num_nodes, num_attrs):
        """Build complete bipartite graph between nodes and attributes"""
        node_idx = torch.arange(num_nodes).repeat_interleave(num_attrs)
        attr_idx = torch.arange(num_attrs).repeat(num_nodes)
        edge_index = torch.stack([node_idx, attr_idx])
        return edge_index 