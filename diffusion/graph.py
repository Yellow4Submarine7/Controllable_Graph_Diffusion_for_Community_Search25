import torch
import torch.nn as nn

class Graph:
    """Graph data structure for diffusion process"""
    def __init__(self, v, e, y=None):
        """
        v: Node features [num_nodes, num_features]
        e: Edge index [2, num_edges] 
        y: Community labels [num_nodes] (optional)
        """
        self.v = v
        self.e = e
        self.y = y
        
    def to(self, device):
        """Move graph to device"""
        self.v = self.v.to(device)
        self.e = self.e.to(device)
        if self.y is not None:
            self.y = self.y.to(device)
        return self
    
    def detach(self):
        """Detach graph tensors"""
        return Graph(
            v=self.v.detach(),
            e=self.e.detach(),
            y=self.y.detach() if self.y is not None else None
        ) 