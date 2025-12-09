from sys import meta_path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import pickle
from load_data import load_data
from tqdm import tqdm
import numpy as np
from Set_GCN_Trans import SetGCN
from Set_GAT_Trans import SetGAT
from SET_GIN_Trans import SetGIN
from Set_GATv2_Trans import SetGATv2
import torch.optim as optim
from sklearn.metrics import r2_score


class ReactionSetDataset(Dataset):
    """
    Each item is one reaction (a set of molecule graphs)
    """
    
    def __init__(self, reactions_list):
        if isinstance(reactions_list, dict):
            self.reactions = list(reactions_list.values())
        else:
            self.reactions = reactions_list
        
    
    def __len__(self):
        return len(self.reactions)
    
    def __getitem__(self, idx):
        return self.reactions[idx]


def collate_reaction_set(batch_list):
    """
    Batch the molecule graphs
    """
    all_molecules = []
    all_yields = []
    reaction_sizes = []

    for reaction in batch_list:
        molecules = reaction['reactants'] + reaction['products']
        all_molecules.extend(molecules)
        all_yields.append(reaction['yield'])
        reaction_sizes.append(len(molecules))
    
    #Batch Molecules
    if len(all_molecules)== 0:
        return None
    
    batched_graphs = Batch.from_data_list(all_molecules)

    #  map each molecule to its reaction and batch them
    reaction_batch = []
    for reaction_idx, size in enumerate(reaction_sizes):
        reaction_batch.extend([reaction_idx] * size)
    reaction_batch = torch.tensor(reaction_batch, dtype=torch.long)

    return {
        'x': batched_graphs.x,
        'edge_index': batched_graphs.edge_index,
        'batch': batched_graphs.batch,
        'yields': torch.tensor(all_yields, dtype=torch.float32),
        'reaction_batch': reaction_batch,
        'num_reactions': len(reaction_sizes)
    }

class SimpleYieldPredictor(nn.Module):
    #Encodes all molecules without distinguishing between reactants and products

    def __init__(self, in_channels, hidden_channels, out_channels, num_gcn_layers=2, mha_dropout=0.2):
        super().__init__()

        self.encoder = SetGAT(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_gcn_layers=num_gcn_layers,
            mha_dropout=mha_dropout
        )

        self.prediction = nn.Sequential(
            nn.Linear(out_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
            

    def forward(self, x, edge_index, batch, reaction_batch):
        # Check input
        if torch.isnan(x).any():
            print(f"NaN in input x!")
            
        molecules_embeddings = self.encoder(x, edge_index, batch)
        
        # Check encoder output
        if torch.isnan(molecules_embeddings).any():
            print(f"NaN after encoder! Mol embs shape: {molecules_embeddings.shape}")
            print(f"First molecule embedding: {molecules_embeddings[0]}")

        # Pool molecuels to reaction level
        num_reactions = reaction_batch.max().item() + 1
        reaction_embeddings = torch.zeros(
            num_reactions,
            molecules_embeddings.size(1),
            device=molecules_embeddings.device
        )
        for i in range(num_reactions):
            mask = (reaction_batch == i)
            reaction_embeddings[i] = molecules_embeddings[mask].mean(dim=0)
        
        # Check after pooling
        if torch.isnan(reaction_embeddings).any():
            print(f"NaN after pooling!")
        
        predictions = self.prediction(reaction_embeddings)
        
        # Check final output
        if torch.isnan(predictions).any():
            print(f"NaN in final predictions!")
            
        return predictions

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    num_batches = 0
    for batch in tqdm(dataloader, desc="Training"):
        if batch is None:
            continue

        x = batch['x'].to(device)
        edge_index = batch['edge_index'].to(device)
        batch_tensor = batch['batch'].to(device)
        reaction_batch = batch['reaction_batch'].to(device)
        yields = batch['yields'].to(device).view(-1, 1)
        
        optimizer.zero_grad()
        predictions = model(x, edge_index, batch_tensor, reaction_batch)
        loss = criterion(predictions, yields)
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"\nWarning: NaN loss detected in batch!")
            print(f"Predictions (first 5): {predictions[:5].detach().cpu()}")
            print(f"Yields (first 5): {yields[:5].cpu()}")
            continue
        
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0
    
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            if batch is None:
                continue

            x = batch['x'].to(device)
            edge_index = batch['edge_index'].to(device)
            batch_tensor = batch['batch'].to(device)
            reaction_batch = batch['reaction_batch'].to(device)
            yields = batch['yields'].to(device).view(-1, 1)

            predictions = model(x, edge_index, batch_tensor, reaction_batch)
            loss = criterion(predictions, yields)
            total_loss += loss.item()
            num_batches += 1

            all_preds.append(predictions.cpu())
            all_targets.append(yields.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate metrics
    mae = torch.abs(all_preds - all_targets).mean().item()
    rmse = torch.sqrt(((all_preds - all_targets) ** 2).mean()).item()
    
    # Calculate R² using scikit-learn
    r2 = r2_score(all_targets.numpy(), all_preds.numpy())
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0

    return avg_loss, mae, rmse, r2
    
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Parameters
    batch_size = 100
    num_epochs = 120
    learning_rate = 1e-3 
    
    # Load data
    #train_data, valid_data, test_data = load_data(
    #    'Data/buchwald_hartwig_processed/train_reactions_graphs.pkl', 
    #    'Data/buchwald_hartwig_processed/valid_reactions_graphs.pkl', 
    #    'Data/buchwald_hartwig_processed/test_reactions_graphs.pkl')
    
    #Load data USPTO

    #train_data, valid_data, test_data = load_data(
    #    'Data/USPTO_data/uspto_train_reactions_graphs.pkl', 
    #    'Data/USPTO_data/uspto_valid_reactions_graphs.pkl', 
    #    'Data/USPTO_data/uspto_test_reactions_graphs.pkl')
    
    train_data, valid_data, test_data = load_data(
        'Data/Drug_Drug/train_reactions_graphs.pkl', 
        'Data/Drug_Drug/valid_reactions_graphs.pkl', 
        'Data/Drug_Drug/test_reactions_graphs.pkl')

    # Dataloaders
    train_loader = DataLoader(
        ReactionSetDataset(train_data),
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_reaction_set,
        num_workers = 0
    )

    valid_loader = DataLoader(
        ReactionSetDataset(valid_data),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_reaction_set,
        num_workers = 0
    )

    test_loader = DataLoader(
        ReactionSetDataset(test_data),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_reaction_set,
        num_workers = 0
    )

    #Get unput dimensions
    first_batch = next(iter(train_loader))
    in_channels = first_batch['x'].shape[1]
    print(f'\nInput features: {in_channels} dimensions')

    #Model
    model = SimpleYieldPredictor(
        in_channels=in_channels,
        hidden_channels = 512,
        out_channels = 1024,
        num_gcn_layers = 6
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)

    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 70)
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae, val_rmse, val_r2 = evaluate(model, valid_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | MAE: {val_mae:.4f} | RMSE: {val_rmse:.4f} | R²: {val_r2:.4f}")
        
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model_level0.pth')
            print("✓ Saved best model")
    
    # Test on best model (AFTER training loop completes)
    print("\n" + "="*70)
    print("Testing best model...")
    print("="*70)
    model.load_state_dict(torch.load('best_model_level0.pth'))
    test_loss, test_mae, test_rmse, test_r2 = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  R²:   {test_r2:.4f}")
    print("="*70)
    
if __name__ == "__main__":
    main()





