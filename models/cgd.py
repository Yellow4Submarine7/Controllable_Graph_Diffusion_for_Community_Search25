import torch
import torch.nn as nn
import torch.nn.functional as F

from .denoising_network import DenoisingNetwork
from .attribute_encoder import AttributeEncoder
from .copilot_network import CopilotNetwork
from ..diffusion.discrete_diffusion import DiscreteDiffusion

class CGD(nn.Module):
    def __init__(self,
                 node_dim,
                 attr_dim,
                 hidden_dim,
                 num_steps=1000,
                 num_layers=6,
                 num_heads=8,
                 dropout=0.2):
        super().__init__()
        
        # Core networks
        self.denoising_net = DenoisingNetwork(
            in_dim=node_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.attr_encoder = AttributeEncoder(
            node_dim=node_dim,
            attr_dim=attr_dim,
            hidden_dim=hidden_dim
        )
        
        # Initialize diffusion process
        self.diffusion = DiscreteDiffusion(
            num_steps=num_steps,
            num_node_classes=node_dim,
            num_community_classes=2
        )
        
        # Copilot network (initialized after base training)
        self.copilot = None
        
    def forward(self, g, query_nodes, query_attrs=None, t=None):
        """
        Forward pass handling both CS and ACS tasks
        """
        if query_attrs is None:
            # Community Search (CS) task
            return self.denoising_net(g, query_nodes, t)
        else:
            # Attributed Community Search (ACS) task
            if self.copilot is None:
                self.copilot = CopilotNetwork(
                    base_model=self.denoising_net,
                    in_dim=g.v.size(-1),
                    hidden_dim=self.denoising_net.layers[0].hidden_dim
                ).to(g.v.device)
                
            # Encode attributes
            enhanced_feats = self.attr_encoder(g, query_attrs)
            
            # Use copilot for ACS
            return self.copilot(g, query_nodes, enhanced_feats, t)
            
    def train_step(self, batch, optimizer):
        """Single training step"""
        optimizer.zero_grad()
        
        # Sample timestep
        t = torch.randint(0, self.diffusion.num_steps, (1,)).item()
        
        # Add noise
        noisy_g = self.diffusion.add_noise(batch.g, t)
        
        # Forward pass
        v_pred, y_pred = self(
            noisy_g, 
            batch.query_nodes,
            batch.query_attrs if hasattr(batch, 'query_attrs') else None,
            t
        )
        
        # Calculate loss
        v_loss = F.cross_entropy(v_pred, batch.g.v)
        y_loss = F.cross_entropy(y_pred, batch.g.y)
        loss = v_loss + y_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return {
            'loss': loss.item(),
            'v_loss': v_loss.item(),
            'y_loss': y_loss.item()
        }
        
    @torch.no_grad()
    def inference(self, g, query_nodes, query_attrs=None):
        """Inference for community search"""
        device = g.v.device
        
        # Initialize with noise
        curr_g = self.diffusion.add_noise(
            g,
            self.diffusion.num_steps-1
        )
        
        # Iterative denoising
        for t in reversed(range(self.diffusion.num_steps)):
            # Model prediction
            curr_g = self.diffusion.sample(
                curr_g,
                lambda g_t, t_: self(g_t, query_nodes, query_attrs, t_),
                t
            )
            
        # Get community predictions
        _, y_pred = self(curr_g, query_nodes, query_attrs, 0)
        communities = torch.argmax(y_pred, dim=-1)
        
        return communities 