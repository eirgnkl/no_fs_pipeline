import pandas as pd
from snakemake.utils import Paramspace
from scripts.utils import create_tasks_df
from pprint import pprint
import numpy as np
import os
import time

# Generate tasks DataFrame and load configuration
os.makedirs("data", exist_ok=True)
tasks_df = create_tasks_df('config.yaml', save='data/tasks.tsv')
time.sleep(5)

tasks_df = pd.read_csv('data/tasks.tsv', sep='\t')

# Extract unique task details
hashes = tasks_df['hash'].unique()
methods = tasks_df['method'].unique()
tasks = tasks_df['task'].unique()

rule all:
    input:
        # Accuracy reports
        expand("data/reports/{task}/{method}/{hash}/accuracy.tsv",
               zip,
               task=tasks_df['task'].str.strip().tolist(),
               method=tasks_df['method'].str.strip().tolist(),
               hash=tasks_df['hash'].str.strip().tolist()),
        # Merge for each task
        expand("data/reports/{task}/merged_results.tsv",
               task=[t.strip() for t in tasks_df['task'].unique()]),
        # Per model best RMSE and R2, and top overall best
        expand("data/reports/{task}/best_results_per_model_rmse.tsv",
               task=[t.strip() for t in tasks_df['task'].unique()]),
        expand("data/reports/{task}/best_results_per_model_r2.tsv",
               task=[t.strip() for t in tasks_df['task'].unique()]),
        expand("data/reports/{task}/best_results_overall_rmse.tsv",
               task=[t.strip() for t in tasks_df['task'].unique()]),
        expand("data/reports/{task}/best_results_overall_r2.tsv",
               task=[t.strip() for t in tasks_df['task'].unique()])

rule run_method:
    input:
        rna_ds=lambda wildcards: tasks_df.loc[tasks_df['hash'] == wildcards.hash, 'input_rna'].values[0],
        metab_ds=lambda wildcards: tasks_df.loc[tasks_df['hash'] == wildcards.hash, 'input_metabolomics'].values[0]
    output:
        metrics="data/reports/{task}/{method}/{hash}/accuracy.tsv",
        predictions="data/reports/{task}/{method}/{hash}/predictions.tsv"
    params:
        thisparam=lambda wildcards: tasks_df.loc[tasks_df['hash'] == wildcards.hash, :].iloc[0, :].to_dict()
    script:
        'scripts/run_method.py'

rule merge:
    input:
        lambda wildcards: expand(
            "data/reports/{task}/{method}/{hash}/accuracy.tsv",
            zip,
            task=[wildcards.task] * len(tasks_df[tasks_df['task'] == wildcards.task]),
            method=tasks_df[tasks_df['task'] == wildcards.task]['method'].tolist(),
            hash=tasks_df[tasks_df['task'] == wildcards.task]['hash'].tolist()
        )
    output:
        tsv="data/reports/{task}/merged_results.tsv"
    run:
        dfs = [pd.read_csv(file, sep='\t') for file in input if os.path.exists(file)]
        merged_df = pd.concat(dfs)
        merged_df.to_csv(output.tsv, sep='\t', index=False)

rule find_best:
    input:
        tsv="data/reports/{task}/merged_results.tsv"
    output:
        per_model_rmse="data/reports/{task}/best_results_per_model_rmse.tsv",
        per_model_r2="data/reports/{task}/best_results_per_model_r2.tsv",
        overall_rmse="data/reports/{task}/best_results_overall_rmse.tsv",
        overall_r2="data/reports/{task}/best_results_overall_r2.tsv"
    run:
        import pandas as pd
        df = pd.read_csv(input.tsv, sep='\t')
        
        # -- Per-model Best Selection --
        per_model_rmse_rows = []
        per_model_r2_rows = []
        grouped = df.groupby(['method_name', 'task'])
        
        for (method_name, task), group in grouped:
            if 'rmse' in group.columns:
                best_row_rmse = group.loc[group['rmse'].idxmin()]
                per_model_rmse_rows.append(best_row_rmse)
            if 'r2' in group.columns:
                best_row_r2 = group.loc[group['r2'].idxmax()]
                per_model_r2_rows.append(best_row_r2)
        
        best_per_model_rmse_df = pd.DataFrame(per_model_rmse_rows).drop_duplicates()
        best_per_model_r2_df = pd.DataFrame(per_model_r2_rows).drop_duplicates()
        best_per_model_rmse_df = best_per_model_rmse_df.sort_values(by='rmse', ascending=True)
        best_per_model_r2_df = best_per_model_r2_df.sort_values(by='r2', ascending=False)

        # -- Overall Top 10 Selection --
        best_overall_rmse_df = df.sort_values(by='rmse', ascending=True).head(10)
        best_overall_r2_df = df.sort_values(by='r2', ascending=False).head(10)
        
        best_per_model_rmse_df.to_csv(output.per_model_rmse, sep='\t', index=False)
        best_per_model_r2_df.to_csv(output.per_model_r2, sep='\t', index=False)
        best_overall_rmse_df.to_csv(output.overall_rmse, sep='\t', index=False)
        best_overall_r2_df.to_csv(output.overall_r2, sep='\t', index=False)
