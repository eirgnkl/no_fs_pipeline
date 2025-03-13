import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error
from scipy.sparse import issparse

def convert_to_tensor(X):
    # If X is sparse, convert to dense
    if issparse(X):
        X = X.toarray()
    # If it's not a numpy array, try to convert it to one
    if not isinstance(X, np.ndarray):
        # If it's already a torch tensor, move it to CPU and convert to numpy
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        else:
            X = np.array(X)
    return torch.tensor(X, dtype=torch.float32)

class MultiLayerGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.5):
        super(MultiLayerGCN, self).__init__()
        
        # Define GCN layers
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))  # First layer
        
        # Add hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))  # Middle layers
        
        self.output_layer = GCNConv(hidden_dim, output_dim)  # Output layer
        
        self.dropout = dropout

    def forward(self, x, edge_index):
        # Pass data through each GCN layer with ReLU activation
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final GCN layer (output layer)
        x = self.output_layer(x, edge_index)
        return x  


def run_gnn(adata_rna_train,
             adata_rna_test,
             adata_msi_train,
             adata_msi_test,
             params,
             **kwargs):
    
    # --- Feature selection ---
    X_train = adata_rna_train.X  
    X_test = adata_rna_test.X  
    Y_train, Y_test = adata_msi_train.X, adata_msi_test.X

    # --- Convert to torch tensors if necessary ---
    X_train = convert_to_tensor(X_train)
    X_test = convert_to_tensor(X_test)
    Y_train = convert_to_tensor(Y_train)
    Y_test = convert_to_tensor(Y_test)

    # --- Device Setup: use GPU if available ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)

    # --- Hyperparameters ---
    hidden_dim = int(params.get('hidden_dim', 256))
    lr = float(params.get('lr', 0.001))
    num_layers = int(params.get('layers', 3))
    dropout = float(params.get('dropout', 0.5))
    k_neighbors = int(params.get('neighbours', 15))
    epochs = int(params.get('epochs', 2000))

    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]

    # --- Initialize model and optimizer ---
    model = MultiLayerGCN(X_train.shape[1], hidden_dim, Y_train.shape[1], num_layers=num_layers, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # --- Training loop ---
    model.train()
    dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_recon, mu, logvar = model(x_batch, y_batch)
            # Reconstruction loss
            recon_loss = mse_loss(y_recon, y_batch)
            # KL divergence loss
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # --- Inference ---
        model.eval()
        with torch.no_grad():
            z_sample = torch.randn(X_test.size(0), latent_dim).to(device)
            decoder_input = torch.cat([X_test, z_sample], dim=1)
            Y_pred = model.decoder(decoder_input).cpu().numpy()

        # --- Evaluation metrics ---
        Y_test_np = Y_test.cpu().numpy()
        pearson_corr = pearsonr(Y_pred.flatten(), Y_test_np.flatten())[0]
        spearman_corr = spearmanr(Y_pred.flatten(), Y_test_np.flatten())[0]
        rmse_test = np.sqrt(mean_squared_error(Y_test_np, Y_pred))
        r2_test = r2_score(Y_test_np, Y_pred)
        mae_test = mean_absolute_error(Y_test_np, Y_pred)

        #Save results to a DataFrame
        metrics = pd.DataFrame({
            'rmse': [rmse_test],
            'mae': [mae_test],
            'r2': [r2_test],
            'pearson': [pearson_corr],
            'spearman': [spearman_corr]
        })

        #Add this for interpretability later, check outputs of each model's preds
        predictions = pd.DataFrame({
            'y_true': Y_test_np.flatten(),
            'y_pred': Y_pred.flatten()
        })

    return metrics, predictions