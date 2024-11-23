import torch
import torch.nn as nn
import numpy as np

class TransitionMatrix:
    """Implementation of uniform transition matrix for discrete diffusion"""
    def __init__(self, num_classes, alpha_schedule='linear'):
        self.num_classes = num_classes
        self.alpha_schedule = alpha_schedule
        
    def get_alpha(self, t, T):
        """Get alpha value for timestep t"""
        if self.alpha_schedule == 'linear':
            return 1 - t/T
        elif self.alpha_schedule == 'cosine':
            return np.cos((t/T) * np.pi/2)
        else:
            raise ValueError(f"Unknown alpha schedule: {self.alpha_schedule}")
    
    def get_Qt(self, t, T):
        """Get transition matrix for timestep t"""
        alpha_t = self.get_alpha(t, T)
        
        # Build uniform transition matrix
        Qt = alpha_t * torch.eye(self.num_classes) + \
             (1 - alpha_t) * torch.ones(self.num_classes, self.num_classes) / self.num_classes
            
        return Qt
    
    def get_Qt_bar(self, t, T):
        """Get accumulated transition matrix from 0 to t"""
        Qt_bar = torch.eye(self.num_classes)
        for s in range(t):
            Qt_bar = torch.matmul(Qt_bar, self.get_Qt(s, T))
        return Qt_bar 