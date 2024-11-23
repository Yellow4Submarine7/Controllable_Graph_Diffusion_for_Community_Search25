import torch
import numpy as np
from torch.utils.data import random_split
from utils.data import CommunitySearchDataset
from utils.metrics import CommunityMetrics

def load_data(dataset_name):
    """Load and preprocess dataset"""
    if dataset_name == "cora":
        # Load Cora dataset
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(root='/tmp/Cora', name='Cora')
        data = dataset[0]
        
        # Convert to numpy arrays
        adj_matrix = torch.zeros((data.num_nodes, data.num_nodes))
        adj_matrix[data.edge_index[0], data.edge_index[1]] = 1
        
        return {
            'adj_matrix': adj_matrix.numpy(),
            'node_features': data.x.numpy(),
            'communities': data.y.numpy(),
            'attr_features': None  # No attribute features for basic CS task
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def main():
    # Configuration
    config = {
        'dataset': 'cora',
        'query_type': 'R1N',
        'num_queries': 4,
        'node_dim': 1433,  # Cora feature dimension
        'attr_dim': None,  # No attributes for basic CS
        'hidden_dim': 256,
        'num_steps': 1000,
        'num_layers': 6,
        'num_heads': 8,
        'dropout': 0.2,
        'batch_size': 32,
        'lr': 0.001,
        'epochs': 100,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'save_path': 'checkpoints/cgd_model.pt'
    }
    
    # Load data
    data_dict = load_data(config['dataset'])
    
    # Create dataset
    dataset = CommunitySearchDataset(
        adj_matrix=data_dict['adj_matrix'],
        node_features=data_dict['node_features'],
        communities=data_dict['communities'],
        query_type=config['query_type'],
        num_queries=config['num_queries'],
        attr_features=data_dict['attr_features']
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Train model
    from train import train
    model = train(config, train_dataset, val_dataset)
    
    # Evaluation
    model.eval()
    val_metrics = []
    
    for batch in val_dataset:
        metrics = CommunityMetrics.evaluate_batch(model, batch, config['device'])
        val_metrics.append(metrics)
    
    # Average metrics
    avg_metrics = {}
    for key in val_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in val_metrics])
    
    print("\nFinal Evaluation Results:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main() 