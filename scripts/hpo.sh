#!/bin/bash
# This script is used to find best hyperparameters on the specified dataset.
# The whole routine includes:
# 1. divide the dataset into K folds (exclude test set, assuming a held-out test set exists).
# 2. for every fold combination, we use the same hyperparameters to retrieve the expectation of the performance on the
#    validation set. That is the performance of the hyperparameters we use.
# 3. using bayesian search to find the best hyperparameters on the dataset.

python main.py experiment_name=frozen2classes run_name=hpo mode=cv hpo=optuna

wait
echo "hpo finished."