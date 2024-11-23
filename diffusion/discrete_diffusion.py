import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscreteDiffusion:
    def __init__(self, num_steps, num_node_classes, num_community_classes):
        self.num_steps = num_steps
        self.node_transition = TransitionMatrix(num_node_classes)
        self.community_transition = TransitionMatrix(num_community_classes)
        
    def add_noise(self, g, t):
        """Add noise to graph for timestep t"""
        # Get transition matrices
        Qt_bar_v = self.node_transition.get_Qt_bar(t, self.num_steps)
        Qt_bar_y = self.community_transition.get_Qt_bar(t, self.num_steps)
        
        # Add noise to node features
        noisy_v = torch.matmul(g.v, Qt_bar_v)
        # Add noise to community labels
        noisy_y = torch.matmul(g.y, Qt_bar_y)
        
        # Keep edge structure unchanged
        return Graph(v=noisy_v, e=g.e, y=noisy_y)
    
    def get_posterior(self, v_t, y_t, v_0, y_0, t):
        """Get posterior distribution for denoising"""
        # Get transition matrices
        Qt_v = self.node_transition.get_Qt(t, self.num_steps)
        Qt_y = self.community_transition.get_Qt(t, self.num_steps)
        Qt_bar_v = self.node_transition.get_Qt_bar(t, self.num_steps)
        Qt_bar_y = self.community_transition.get_Qt_bar(t, self.num_steps)
        
        # Calculate posterior for nodes
        v_coeff = Qt_v / Qt_bar_v
        v_posterior = v_coeff * torch.matmul(v_t, Qt_bar_v.inverse())
        
        # Calculate posterior for communities
        y_coeff = Qt_y / Qt_bar_y  
        y_posterior = y_coeff * torch.matmul(y_t, Qt_bar_y.inverse())
        
        return v_posterior, y_posterior
    
    def sample(self, g_t, model, t):
        """Sample from posterior for denoising step"""
        # Get model predictions
        v_pred, y_pred = model(g_t, t)
        
        # Get posterior distributions
        v_posterior, y_posterior = self.get_posterior(
            g_t.v, g_t.y, v_pred, y_pred, t
        )
        
        # Sample from posteriors
        v_sample = torch.multinomial(v_posterior, 1).squeeze()
        y_sample = torch.multinomial(y_posterior, 1).squeeze()
        
        return Graph(v=v_sample, e=g_t.e, y=y_sample) 