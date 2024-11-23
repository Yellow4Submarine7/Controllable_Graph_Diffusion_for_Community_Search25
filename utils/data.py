import torch
from torch.utils.data import Dataset
import numpy as np
from ..diffusion.graph import Graph

class CommunitySearchDataset(Dataset):
    def __init__(self, 
                 adj_matrix,
                 node_features,
                 communities,
                 query_type='R1N',
                 num_queries=4,
                 attr_features=None):
        """
        adj_matrix: Adjacency matrix [num_nodes, num_nodes]
        node_features: Node feature matrix [num_nodes, node_dim]
        communities: Ground truth community labels [num_nodes]
        query_type: Type of query generation ('R1N', 'R5N', 'TAC', 'RAC')
        num_queries: Number of query nodes/attributes to sample
        attr_features: Optional attribute features for ACS tasks
        """
        self.adj_matrix = torch.from_numpy(adj_matrix).float()
        self.node_features = torch.from_numpy(node_features).float()
        self.communities = torch.from_numpy(communities).long()
        self.query_type = query_type
        self.num_queries = num_queries
        self.attr_features = torch.from_numpy(attr_features).float() if attr_features is not None else None
        
        # Convert adjacency matrix to edge index
        self.edge_index = self._adj_to_edge_index(self.adj_matrix)
        
        # Generate queries
        self.queries = self._generate_queries()
        
    def _adj_to_edge_index(self, adj):
        """Convert adjacency matrix to edge index format"""
        edges = torch.nonzero(adj).t()
        return edges
        
    def _generate_queries(self):
        """Generate query sets based on query type"""
        queries = []
        unique_communities = torch.unique(self.communities)
        
        for comm in unique_communities:
            comm_nodes = torch.nonzero(self.communities == comm).squeeze()
            
            if self.query_type == 'R1N':
                # Random 1 node
                query_nodes = comm_nodes[torch.randperm(len(comm_nodes))[:1]]
                query_attrs = None
                
            elif self.query_type == 'R5N':
                # Random 5 nodes
                query_nodes = comm_nodes[torch.randperm(len(comm_nodes))[:5]]
                query_attrs = None
                
            elif self.query_type in ['TAC', 'RAC']:
                # Sample query nodes
                query_nodes = comm_nodes[torch.randperm(len(comm_nodes))[:self.num_queries]]
                
                if self.attr_features is not None:
                    comm_attrs = self.attr_features[comm_nodes]
                    if self.query_type == 'TAC':
                        # Top attributes
                        attr_freq = torch.sum(comm_attrs, dim=0)
                        query_attrs = torch.topk(attr_freq, self.num_queries).indices
                    else:
                        # Random attributes
                        query_attrs = torch.randperm(comm_attrs.size(1))[:self.num_queries]
                else:
                    query_attrs = None
                    
            queries.append({
                'nodes': query_nodes,
                'attrs': query_attrs,
                'community': comm
            })
            
        return queries
    
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        query = self.queries[idx]
        
        # Construct graph object
        g = Graph(
            v=self.node_features,
            e=self.edge_index,
            y=self.communities
        )
        
        return {
            'g': g,
            'query_nodes': query['nodes'],
            'query_attrs': query['attrs'],
            'target_community': query['community']
        } 