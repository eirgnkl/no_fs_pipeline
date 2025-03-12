## 🐍 Pipeline for Metabolite Prediction from Gene Expression

This repository contains a Snakemake pipeline for predicting **metabolic distribution from gene expression data** using several **machine learning models.** For this version of the pipeline **there is no preprocessing** taking place, as it's tedious and msi data preprocessing is not standardized yet. For this, prepare the data as you wish beforehand, store them in desired path and it in the config.yaml.

## 🏗️ Pipeline Structure

The pipeline is structured as follows:

1. **Model Training**: Runs Ridge, Lasso, Linear Regression, XGBoost and a CVAE.
2. **Evaluation**: Assesses performance using 5 different metrics (RMSE, MAE, Pearson and Spearmann Correlation and $R^2$).

## 🔧 How to set Tasks and Models:

Visit `config.yaml` and follow these steps:

1. Set name of `'task'` in `config.yaml` under the `TASKS` key.
2. Set correct paths to your RNA and MSI datasets in `input_rna` and `input_metabolomics.`
3. Define the name of the `split `that you want to use for the task
4. Specify the `methods` (models) that you want to use to make predictions. For the parameter tuning keep reading.
5. Select the different ways of preprocessing that you want your models to run with, under the `featsel` key.

```
TASKS:
  'vitatrack':
    input_rna: /path_to_rna/rna_file.h5ad
    input_metabolomics: /path_to_msi/msi_file.h5ad
    split: split
    methods:
      ridge:
        params: params/ridge_params.tsv
      linear:
        params: params/linreg_params.tsv
```

## 📈 Models Implemented

The pipeline supports the following regression models:

- **Ridge Regression**: Handles multicollinearity by adding an L2 penalty.
- **Lasso Regression**: Adds an L1 penalty for feature selection.
- **Linear Regression**: Standard least squares regression.
- **XGBoost**: Gradient boosting for non-linear patterns.

In case you want to add new models, be aware that the models is called through the `run_methods.py`, so make sure the structure of it is similar to the already existing scripts and define a `{new_methods}_param.tsv`, in the folder `params`

### Hyperparameters to Tune

Each model has parameters that users can configure in `params/{method}_params.tsv`.

## 🏃 Running the Pipeline

To execute the pipeline, use:

```bash
snakemake --cores <num_cores> --profile profile_gpu
```

For dry-run mode:

```bash
snakemake --dry-run
```

## 🌈Visualization

### Model Performance Visualization

After the pipeline completes, a **visualization step** generates comparative plots for each task. These plots provide a clear view of model performance across different **feature selection techniques**. User sets in snakefile the **desired number of best models** to view and compare in the params `rule visualize`

### Visualization Includes:

- 📊 **Bar charts** showing model performance for each metric (**RMSE, Pearson, Spearman, R²**).
- 🎯 **Feature selection methods displayed inside bars** instead of model parameters.
- ⭐ **Best-performing models highlighted** for each metric.

These plots help assess **which model with which parameters and feature selection techniques yield the best results** for each task.

## 🗂️ Output

Results are stored in:

```
data/reports/{TASK}/  # Best results for each task
  ├── best_results.tsv  # Summary of best-performing models
  ├── {model}/{feature_selection}/  # Model-specific results
  │   ├── accuracy.tsv  # Model performance metrics
  │   ├── predictions.tsv  # Predicted metabolites

```

*Alternative to using profile_gpu:*

```
snakemake --jobs 10 --cluster "mkdir -p logs/{rule} && sbatch --partition=gpu_p --gres=gpu:1 --mem=32000 --qos=gpu_normal --job-name=smk-{rule}-{wildcards} --output=logs/{rule}/%j-{rule}-{wildcards}.out --error=logs/{rule}/%j-{rule}-{wildcards}.err --nice=10000 --exclude=supergpu05,supergpu08,supergpu07,supergpu02,supergpu03 --parsable" --cluster-cancel "scancel {cluster_jobid}"
```
