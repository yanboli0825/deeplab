import numpy as np

from src.datamodules.split import KFoldSplitProvider, make_group_kfold_split, make_kfold_split


def test_make_kfold_split_covers_all_samples() -> None:
    """Verify that K-fold splitting covers all samples without overlap.

    Returns:
        None: Assertions validate split coverage and disjointness.
    """

    split = make_kfold_split(num_samples=12, n_splits=3, fold=1, seed=7)

    combined = np.concatenate([split.train, split.val])
    assert sorted(combined.tolist()) == list(range(12))
    assert len(np.intersect1d(split.train, split.val)) == 0


def test_make_group_kfold_split_keeps_group_together() -> None:
    """Verify that group K-fold never splits one group across train and validation.

    Returns:
        None: Assertions validate group isolation.
    """

    group_ids = [0, 0, 1, 1, 2, 2, 3, 3]
    split = make_group_kfold_split(
        num_samples=len(group_ids),
        group_ids=group_ids,
        n_splits=4,
        fold=2,
        seed=11,
    )

    val_groups = {group_ids[idx] for idx in split.val.tolist()}
    train_groups = {group_ids[idx] for idx in split.train.tolist()}
    assert val_groups.isdisjoint(train_groups)


def test_kfold_split_provider_builds_requested_fold() -> None:
    """Verify that the split provider produces one valid requested fold.

    Returns:
        None: Assertions validate provider output shape and disjointness.
    """

    provider = KFoldSplitProvider(num_samples=10, n_splits=5, fold=3, seed=5)
    split = provider.build()

    assert len(split.val) > 0
    assert len(np.intersect1d(split.train, split.val)) == 0
