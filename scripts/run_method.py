import scanpy as sc
import pandas as pd
import ast

from ridge import run_ridge_reg
from linear import run_linreg
from lasso import run_lasso
from mxgboost import run_xgboost
from elastic_net import run_elastic_net
from cvae import run_cvae

# Mapping method names to their respective functions
METHOD_MAP = {
    'ridge': dict(function=run_ridge_reg, mode='paired'),
    'lasso': dict(function=run_lasso, mode='paired'),
    'linear': dict(function=run_linreg, mode='paired'),
    'xgboost': dict(function=run_xgboost, mode='paired'),
    'elastic_net': dict(function=run_elastic_net, mode='paired'),
    'cvae': dict(function=run_cvae, mode='paired')
}

# Load parameters from Snakemake
params = snakemake.params.thisparam 
rna_path = snakemake.input.rna_ds
metab_path = snakemake.input.metab_ds

# Task parameters
method = params['method']
task = params['task']
hash_id = params['hash']
split_name = params.get("split", "split")

method_params = ast.literal_eval(params['params'])
method_mode = METHOD_MAP[method]['mode']
method_function = METHOD_MAP[method]['function']

adata_rna = sc.read_h5ad(rna_path)
adata_msi = sc.read_h5ad(metab_path)

adata_rna_train = adata_rna[adata_rna.obs[split_name] == "train", :]
adata_rna_test  = adata_rna[adata_rna.obs[split_name] == "test", :]

adata_msi_train = adata_msi[adata_msi.obs[split_name] == "train", :]
adata_msi_test  = adata_msi[adata_msi.obs[split_name] == "test", :]

metrics_df, predictions_df = method_function(
    adata_rna_train=adata_rna_train,
    adata_rna_test=adata_rna_test,
    adata_msi_train=adata_msi_train,
    adata_msi_test=adata_msi_test, 
    params=method_params
)

# Add metadata to the results and save to output files
metrics_df['task'] = task
metrics_df['method_name'] = method
metrics_df['method_params'] = str(method_params)
metrics_df['hash'] = hash_id
metrics_df.to_csv(snakemake.output.metrics, sep='\t', index=False)
predictions_df.to_csv(snakemake.output.predictions, sep='\t', index=False)
