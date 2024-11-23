import torch
import torch.nn as nn
from ..layers.zero_gcn import ZeroGCN

class CopilotNetwork(nn.Module):
    def __init__(self, base_model, in_dim, hidden_dim):
        super().__init__()
        
        # Copy and freeze base model
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Zero-initialized GCN layers
        self.attr_gcn = ZeroGCN(in_dim, hidden_dim)
        self.fusion_gcn = ZeroGCN(hidden_dim, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, in_dim)
        
    def forward(self, g, query_nodes, query_attrs, t):
        """
        g: Graph object
        query_nodes: Query node indices
        query_attrs: Enhanced node features from attribute encoder
        t: Current timestep
        """
        # Get base model predictions
        base_node_pred, base_comm_pred = self.base_model(g, query_nodes, t)
        
        # Process attribute information
        attr_feats = self.attr_gcn(query_attrs, g.e)
        
        # Fuse base predictions with attribute features
        fused_feats = self.fusion_gcn(base_node_pred + attr_feats, g.e)
        
        # Generate final predictions
        node_pred = self.output_proj(fused_feats)
        community_pred = base_comm_pred  # Keep original community predictions
        
        return node_pred, community_pred 