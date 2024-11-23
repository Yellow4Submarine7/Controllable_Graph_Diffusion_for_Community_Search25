import torch
import torch.nn as nn
import torch.nn.functional as F

class QueryDrivenGraphTransformer(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads=8, dropout=0.2):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Multi-head attention layers
        self.q_linear = nn.Linear(in_dim, hidden_dim * num_heads)
        self.k_linear = nn.Linear(in_dim, hidden_dim * num_heads)
        self.v_linear = nn.Linear(in_dim, hidden_dim * num_heads)
        
        # FiLM layer for time step conditioning
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2)
        )
        
        # Output layers
        self.out_linear = nn.Linear(hidden_dim * num_heads, in_dim)
        self.layer_norm = nn.LayerNorm(in_dim)
        
    def forward(self, x, edge_index, query_nodes, t):
        batch_size = x.size(0)
        
        # Reshape for multi-head attention
        q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.hidden_dim)
        k = self.k_linear(x).view(batch_size, -1, self.num_heads, self.hidden_dim)
        v = self.v_linear(x).view(batch_size, -1, self.num_heads, self.hidden_dim)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
        
        # Mask for query nodes
        query_mask = torch.zeros_like(scores)
        query_mask[:, query_nodes, :] = float('-inf')
        scores = scores + query_mask
        
        # Apply attention
        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        
        # Time step conditioning
        time_emb = self.time_mlp(t.unsqueeze(-1))
        gamma, beta = time_emb.chunk(2, dim=-1)
        
        # Apply FiLM conditioning
        out = torch.matmul(attn, v)
        out = out * (1 + gamma) + beta
        
        # Output projection
        out = self.out_linear(out.view(batch_size, -1, self.hidden_dim * self.num_heads))
        out = self.layer_norm(out + x)  # Residual connection
        
        return out 