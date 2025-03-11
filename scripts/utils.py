import pandas as pd
import yaml
import hashlib
import os
import numpy as np
from scipy.spatial import cKDTree

def create_hash(string: str, digest_size: int = 5):
    string = string.encode('utf-8')
    return hashlib.blake2b(string, digest_size=digest_size).hexdigest()

def create_tasks_df(config, save=None):
    tasks_df = []
    with open(config, "r") as stream:
        params = yaml.safe_load(stream)
    
    for task in params['TASKS']:
        task_dict = params['TASKS'][task]
        method_dfs = []
        
        for method, method_data in task_dict['methods'].items():
            # If method_data is a string, it’s the parameters file path;
            # if a dict, we extract the parameters file path.
            if isinstance(method_data, str):
                method_params = method_data
            elif isinstance(method_data, dict):
                method_params = method_data.get('params')
            else:
                raise ValueError(f"Unexpected format for method_data: {method_data}")
            
            if method_params:
                df_params = pd.read_csv(method_params, sep='\t', index_col=0)
                params_list = [str(row) for row in df_params.to_dict(orient='records')]
            else:
                df_params = pd.DataFrame()
                params_list = [{}]
            
            # Create rows for the method (feature selection removed)
            method_df = {
                'params': params_list,
                'hash': [create_hash(row + method + task) for row in params_list],
                'method': [method] * len(params_list),
            }
            method_dfs.append(pd.DataFrame(method_df))
        
        if method_dfs:
            method_dfs = pd.concat(method_dfs, ignore_index=True)
            method_dfs['task'] = task

            # Add any additional task-level attributes
            for key in task_dict:
                if key != 'methods':
                    method_dfs[key] = task_dict[key]
            
            tasks_df.append(method_dfs)
    
    if tasks_df:
        tasks_df = pd.concat(tasks_df, ignore_index=True)
    else:
        tasks_df = pd.DataFrame()
    
    if save is not None:
        tasks_df.to_csv(save, sep='\t', index=False)
    
    return tasks_df
