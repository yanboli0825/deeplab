import numpy as np

from src.datamodules.split import make_group_kfold_split, make_kfold_split


def test_make_kfold_split_covers_all_samples() -> None:
    split = make_kfold_split(num_samples=12, n_splits=3, fold=1, seed=7)

    combined = np.concatenate([split.train, split.val])
    assert sorted(combined.tolist()) == list(range(12))
    assert len(np.intersect1d(split.train, split.val)) == 0


def test_make_group_kfold_split_keeps_group_together() -> None:
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
