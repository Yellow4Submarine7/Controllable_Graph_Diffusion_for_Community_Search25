import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.cgd import CGD

def train(config, train_dataset, val_dataset):
    # Initialize model
    model = CGD(
        node_dim=config.node_dim,
        attr_dim=config.attr_dim,
        hidden_dim=config.hidden_dim,
        num_steps=config.num_steps,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout
    ).to(config.device)
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config.epochs):
        # Training
        model.train()
        train_metrics = []
        for batch in train_loader:
            batch = batch.to(config.device)
            metrics = model.train_step(batch, optimizer)
            train_metrics.append(metrics)
            
        # Validation
        model.eval()
        val_metrics = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(config.device)
                v_pred, y_pred = model(
                    batch.g,
                    batch.query_nodes,
                    getattr(batch, 'query_attrs', None),
                    0
                )
                
                v_loss = F.cross_entropy(v_pred, batch.g.v)
                y_loss = F.cross_entropy(y_pred, batch.g.y)
                loss = v_loss + y_loss
                
                val_metrics.append({
                    'loss': loss.item(),
                    'v_loss': v_loss.item(),
                    'y_loss': y_loss.item()
                })
                
        # Log metrics
        train_results = {k: sum(m[k] for m in train_metrics)/len(train_metrics) 
                        for k in train_metrics[0]}
        val_results = {k: sum(m[k] for m in val_metrics)/len(val_metrics)
                      for k in val_metrics[0]}
        
        print(f"Epoch {epoch}:")
        print(f"Train: {train_results}")
        print(f"Val: {val_results}")
        
        # Save best model
        if val_results['loss'] < best_val_loss:
            best_val_loss = val_results['loss']
            torch.save(model.state_dict(), config.save_path)
            
    return model 