import pandas as pd
import scanpy as sc
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.metrics import mean_absolute_error

from scipy.stats import spearmanr, pearsonr
import numpy as np
from scipy.sparse import issparse

def convert_to_dense(matrix):
    """Converts a sparse matrix to dense if necessary."""
    if issparse(matrix):
        return matrix.toarray()
    return matrix

def run_linreg(
        adata_rna_train,
        adata_rna_test,
        adata_msi_train,
        adata_msi_test, 
        params, 
        # featsel,
        **kwargs):

    #adding feature selection as a param to select correct parts of the adata
    # if featsel == "hvg":
    X_train = adata_rna_train.X  
    X_test = adata_rna_test.X  
    Y_train, Y_test = adata_msi_train.X, adata_msi_test.X
   
    # Convert to dense if needed
    X_train = convert_to_dense(X_train)
    X_test = convert_to_dense(X_test)
    Y_train = convert_to_dense(Y_train)
    Y_test = convert_to_dense(Y_test)

    # Fit linear regression
    lin = LinearRegression()
    lin.fit(X_train, Y_train)

    # Predictions and evaluation
    Y_pred = lin.predict(X_test)
    
    #Pearson spearman
    pearson_corr = pearsonr(Y_pred.flatten(), Y_test.flatten())[0]
    spearman_corr = spearmanr(Y_pred.flatten(), Y_test.flatten())[0]

    #MSE and R2
    rmse_test = root_mean_squared_error(Y_test, Y_pred)
    r2_test = r2_score(Y_test, Y_pred)
    mae_test = mean_absolute_error(Y_test, Y_pred)

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
        'y_true': Y_test.flatten(),
        'y_pred': Y_pred.flatten()
    })

    return metrics, predictions
