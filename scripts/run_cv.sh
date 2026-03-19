#!/bin/bash
# assume that:
# 1. we have found the best hyperparameter combination via cross validation on validation set.
# 2. our dataset is not large enough so that we can't get a held-out test set. To better utilize the data, we can also
#    divide the data into K folds, then retrieve the expectation of the performance on the test set. Same as cross validation.

# parallelly start the tasks
python main.py experiment_name=performance_test run_name=fold0 model.optimizer.lr=0.0005
python main.py experiment_name=performance_test run_name=fold1 model.optimizer.lr=0.0005
python main.py experiment_name=performance_test run_name=fold2 model.optimizer.lr=0.0005
python main.py experiment_name=performance_test run_name=fold3 model.optimizer.lr=0.0005
python main.py experiment_name=performance_test run_name=fold5 model.optimizer.lr=0.0005

wait
echo "All 5 folds finished."