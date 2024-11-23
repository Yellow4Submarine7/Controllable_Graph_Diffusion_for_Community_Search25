import torch
import networkx as nx
import matplotlib.pyplot as plt
from utils.data import CommunitySearchDataset
from models.cgd import CGD

def visualize_community(g, communities, query_nodes, title):
    """Visualize detected community"""
    # Create networkx graph
    G = nx.Graph()
    edge_list = g.e.t().tolist()
    G.add_edges_from(edge_list)
    
    # Set node colors
    colors = ['red' if communities[i] == 1 else 'lightgray' for i in range(len(communities))]
    for node in query_nodes:
        colors[node] = 'yellow'
    
    # Draw graph
    plt.figure(figsize=(10, 10))
    nx.draw(G, node_color=colors, with_labels=True, node_size=500)
    plt.title(title)
    plt.show()

def main():
    # Load trained model
    config = torch.load('checkpoints/cgd_model.pt')
    model = CGD(**config['model_params'])
    model.load_state_dict(config['state_dict'])
    model = model.to(config['device'])
    model.eval()
    
    # Load test data
    data_dict = load_data(config['dataset'])
    dataset = CommunitySearchDataset(
        adj_matrix=data_dict['adj_matrix'],
        node_features=data_dict['node_features'],
        communities=data_dict['communities'],
        query_type='R1N',
        num_queries=1
    )
    
    # Run inference on a sample
    sample = dataset[0]
    g = sample['g'].to(config['device'])
    query_nodes = sample['query_nodes'].to(config['device'])
    
    # Get predictions
    with torch.no_grad():
        pred_communities = model.inference(g, query_nodes)
    
    # Visualize results
    visualize_community(
        g,
        pred_communities.cpu(),
        query_nodes.cpu(),
        "Detected Community"
    )
    
    # Print metrics
    true_communities = g.y
    metrics = CommunityMetrics.calculate_metrics(pred_communities, true_communities)
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main() 