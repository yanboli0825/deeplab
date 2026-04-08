# Module Guide

## Split Module

### Summary

The split entry surface is intentionally small:

- `split provider`
- datamodule `_default_split()`

Public split entry points such as `split_indices` and `split_file` are no longer part of the framework contract.

### Current Contract

Runtime behavior is:

1. build a `split provider` from the top-level `split` config
2. inject the provider into the datamodule as a runtime object
3. let the datamodule call `provider.build()` inside `resolve_split()`
4. fall back to `_default_split()` only if no provider is supplied

The datamodule owns data consumption, not split policy.

### Supported Split Families

The current framework supports four split families:

- non-stratified
- group-aware
- stratified
- stratified + group-aware

Representative methods include:

- `holdout`
- `train_val_test_holdout`
- `kfold`
- `dev_test_kfold`
- `group_holdout`
- `group_train_val_test_holdout`
- `group_kfold`
- `group_dev_test_kfold`
- `stratified_holdout`
- `stratified_train_val_test_holdout`
- `stratified_kfold`
- `stratified_dev_test_kfold`
- `stratified_group_holdout`
- `stratified_group_train_val_test_holdout`
- `stratified_group_kfold`
- `stratified_group_dev_test_kfold`

### Group-Aware Split

For group-aware split, the recommended config is:

- `split.data_file`
- `split.group_id_column`

The framework resolves metadata from the manifest during `build_split_provider()`:

- sample count
- group ids

Users do not need to hand-write:

- `num_samples`
- `group_ids`
- `candidate_indices`

### Stratified Split

For stratified split, the recommended config is:

- `split.data_file`
- `split.label_column`

The framework resolves labels from the manifest during `build_split_provider()`.

### Stratified Group Split Semantics

`stratified_group_*` is not “random group split with a little stratification”. The intended semantics are:

- `group` non-leakage is a hard constraint
- every split must contain all classes is a hard constraint
- label distribution should stay close to the global distribution
- bin size balance is a secondary objective

If the data cannot satisfy these constraints under the requested `n_folds` or `test_ratio`, the split stage should fail early with a clear error.

This is important in class-imbalanced medical tasks, where having many samples does not imply that every class is available in enough unique groups.
