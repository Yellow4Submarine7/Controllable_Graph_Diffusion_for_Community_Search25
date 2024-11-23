import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class CommunityMetrics:
    """Evaluation metrics for community search"""
    
    @staticmethod
    def calculate_metrics(pred_communities, true_communities):
        """
        Calculate precision, recall, F1 and accuracy
        
        pred_communities: Predicted community assignments [num_nodes]
        true_communities: Ground truth community labels [num_nodes]
        """
        # Convert to numpy for sklearn metrics
        if torch.is_tensor(pred_communities):
            pred_communities = pred_communities.cpu().numpy()
        if torch.is_tensor(true_communities):
            true_communities = true_communities.cpu().numpy()
            
        # Calculate metrics
        precision = precision_score(true_communities, pred_communities, average='macro')
        recall = recall_score(true_communities, pred_communities, average='macro')
        f1 = f1_score(true_communities, pred_communities, average='macro')
        accuracy = accuracy_score(true_communities, pred_communities)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        }
    
    @staticmethod
    def evaluate_batch(model, batch, device):
        """Evaluate model predictions on a batch"""
        batch = {k: v.to(device) if torch.is_tensor(v) else v 
                for k, v in batch.items()}
        
        # Get predictions
        pred_communities = model.inference(
            batch['g'],
            batch['query_nodes'],
            batch.get('query_attrs', None)
        )
        
        # Calculate metrics
        metrics = CommunityMetrics.calculate_metrics(
            pred_communities,
            batch['g'].y
        )
        
        return metrics 