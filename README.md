# PdM - Machine Failure Prediction with GBDTs

Predictive maintanence (PdM) is the maintanence of machines at a predicted future time before the machine failure. This allows scheduled maintanence of the machines, reducing the unplanned downtime costs.

This repository contains deployable end-to-end classifier models to predit the probability whether a machine failure will occur or not. The models include state-of-the-art gradient boosted decision tree models: xgboost, lightgbm, and catboost.

## Data

The data used in this work, which is taken from [Matzka (2020)](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset), is available in the DATA folder. It consists of 10,000 data points stored as rows with features like product type, air temperature, process temperature, rotational speed, torque wear, machine failure. The machine failures are grouped into 5 subcategories. For simplicity, only the subcategories were not predicted. A concise summary of the data, pair plots, and the distribution of the labels can be found in [this](./EDA/EDA.ipynb) notebook.

## Modeling

The modeling was done with the sklearn api and sklearn api of the xgboost, lightgbm, and catboost libraries. The modeling process consists of a parameter optimization part and a testing part. The optimal model parameters were obtained with the optuna library.

### Parameter Optimization

Run the the following command in MODELS folder to tune the xgboost model in 20 trials.

```shell
python3 tuner.py -est-name xgb -nt 20
```

All trials are stored and their corresponding scores are stored in study_{est_name}.pkl. files.

### Model Testing

After parameter optimization, the models can be re-trained and saved by runing the following command:

```shell
python3 tester.py
```

The command saves two kinds of files: models with file names {est-name}.model (e.g. xgb.model) and oof predictions with file names {est-name}_oof_preds.npy (e.g. xgb_oof_preds.npy).

## Requirements

The [requirements.txt](requirements.txt) file lists all the python libraries needed to run the codes in this repository, and they can be installed using:

```shell
pip install -r requirements.txt
```


