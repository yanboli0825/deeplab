import csv

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.datamodules.split import (
    make_stratified_dev_test_kfold_split,
    make_stratified_group_dev_test_holdout_split,
    make_stratified_group_dev_test_kfold_split,
    make_stratified_group_holdout_split,
    make_stratified_group_kfold_split,
    make_stratified_holdout_split,
    make_stratified_kfold_split,
)
from src.utils.build import build_split_provider


def test_stratified_holdout_preserves_binary_label_ratio() -> None:
    labels = np.array([0] * 80 + [1] * 20)
    split = make_stratified_holdout_split(
        num_samples=len(labels),
        labels=labels,
        val_ratio=0.25,
        seed=7,
    )

    val_labels = labels[split.val]
    train_labels = labels[split.train]

    assert len(split.val) == 25
    assert int(val_labels.sum()) == 5
    assert int(train_labels.sum()) == 15


def test_stratified_kfold_balances_class_counts_across_folds() -> None:
    labels = np.array([0] * 80 + [1] * 20)
    positives_per_fold = []

    for fold in range(5):
        split = make_stratified_kfold_split(
            num_samples=len(labels),
            labels=labels,
            n_splits=5,
            fold=fold,
            seed=11,
        )
        positives_per_fold.append(int(labels[split.val].sum()))

    assert positives_per_fold == [4, 4, 4, 4, 4]


def test_stratified_group_dev_test_kfold_keeps_test_fixed_and_groups_isolated() -> None:
    labels = np.array([0, 0, 1, 1] * 6)
    group_ids = np.array(
        [f"patient_{idx}" for idx in range(12) for _ in range(2)],
        dtype=object,
    )

    split_fold0 = make_stratified_group_dev_test_kfold_split(
        num_samples=len(labels),
        group_ids=group_ids,
        labels=labels,
        n_splits=3,
        fold=0,
        test_ratio=0.25,
        seed=13,
    )
    split_fold1 = make_stratified_group_dev_test_kfold_split(
        num_samples=len(labels),
        group_ids=group_ids,
        labels=labels,
        n_splits=3,
        fold=1,
        test_ratio=0.25,
        seed=13,
    )

    assert set(split_fold0.test.tolist()) == set(split_fold1.test.tolist())

    train_groups = set(group_ids[split_fold0.train].tolist())
    val_groups = set(group_ids[split_fold0.val].tolist())
    test_groups = set(group_ids[split_fold0.test].tolist())
    train_labels = set(labels[split_fold0.train].tolist())
    val_labels = set(labels[split_fold0.val].tolist())
    test_labels = set(labels[split_fold0.test].tolist())

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)
    assert train_labels == {0, 1}
    assert val_labels == {0, 1}
    assert test_labels == {0, 1}


def test_build_split_provider_reads_label_and_group_columns_from_data_file(tmp_path) -> None:
    data_file = tmp_path / "manifest.csv"
    with open(data_file, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "label", "patient_id"])
        writer.writeheader()
        for idx in range(12):
            writer.writerow(
                {
                    "sample_id": f"s{idx}",
                    "label": "pos" if idx % 3 == 0 else "neg",
                    "patient_id": f"p{idx // 2}",
                }
            )

    provider = build_split_provider(
        OmegaConf.create(
            {
            "method": "stratified_group_dev_test_kfold",
            "data_file": str(data_file),
            "group_id_column": "patient_id",
            "label_column": "label",
            "n_folds": 3,
            "test_ratio": 0.25,
            "seed": 19,
            }
        ),
        fold=0,
    )

    split = provider.build()
    assert len(split.test) > 0
    assert len(split.val) > 0
    assert len(split.train) > 0


def test_stratified_group_kfold_keeps_validation_non_empty_under_uneven_groups() -> None:
    labels = np.array([1] * 8 + [0] * 6 + [1, 0, 0, 1, 0, 0, 0, 1])
    group_ids = np.array(
        ["g0"] * 8 + ["g1"] * 6 + ["g2"] * 2 + ["g3"] * 2 + ["g4"] * 2 + ["g5"] * 2,
        dtype=object,
    )

    for fold in range(3):
        split = make_stratified_group_kfold_split(
            num_samples=len(labels),
            group_ids=group_ids,
            labels=labels,
            n_splits=3,
            fold=fold,
            seed=23,
        )
        assert len(split.val) > 0
        train_groups = set(group_ids[split.train].tolist())
        val_groups = set(group_ids[split.val].tolist())
        train_labels = set(labels[split.train].tolist())
        val_labels = set(labels[split.val].tolist())
        assert train_groups.isdisjoint(val_groups)
        assert train_labels == {0, 1}
        assert val_labels == {0, 1}


def test_stratified_group_kfold_requires_enough_unique_groups() -> None:
    labels = np.array([0, 0, 1, 1, 0, 1])
    group_ids = np.array(["a", "a", "b", "b", "c", "c"], dtype=object)

    with pytest.raises(ValueError, match="requires at least n_splits unique groups"):
        make_stratified_group_kfold_split(
            num_samples=len(labels),
            group_ids=group_ids,
            labels=labels,
            n_splits=4,
            fold=0,
            seed=31,
        )


def test_stratified_group_holdout_requires_each_class_in_two_groups() -> None:
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    group_ids = np.array(["p0", "p0", "p1", "p1", "p2", "p2", "p2", "p2"], dtype=object)

    with pytest.raises(ValueError, match="requires each class to appear in at least 2 unique groups"):
        make_stratified_group_holdout_split(
            num_samples=len(labels),
            group_ids=group_ids,
            labels=labels,
            val_ratio=0.5,
            seed=37,
        )


def test_stratified_group_dev_test_holdout_produces_both_classes_on_each_side() -> None:
    labels = np.array([0, 1] * 8)
    group_ids = np.array([f"patient_{idx}" for idx in range(8) for _ in range(2)], dtype=object)

    split = make_stratified_group_dev_test_holdout_split(
        num_samples=len(labels),
        group_ids=group_ids,
        labels=labels,
        test_ratio=0.25,
        seed=41,
    )

    assert set(labels[split.train].tolist()) == {0, 1}
    assert set(labels[split.test].tolist()) == {0, 1}


def test_stratified_group_dev_test_kfold_fails_when_dev_lacks_class_group_coverage() -> None:
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    group_ids = np.array(["p0", "p0", "p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"], dtype=object)

    with pytest.raises(ValueError, match="dev/test K-fold \\(dev partition\\) requires each class to appear"):
        make_stratified_group_dev_test_kfold_split(
            num_samples=len(labels),
            group_ids=group_ids,
            labels=labels,
            n_splits=2,
            fold=0,
            test_ratio=0.4,
            seed=43,
        )


def test_stratified_dev_test_kfold_keeps_test_fixed_across_folds() -> None:
    labels = np.array([0] * 80 + [1] * 20)

    split_fold0 = make_stratified_dev_test_kfold_split(
        num_samples=len(labels),
        labels=labels,
        n_splits=5,
        fold=0,
        test_ratio=0.2,
        seed=5,
    )
    split_fold1 = make_stratified_dev_test_kfold_split(
        num_samples=len(labels),
        labels=labels,
        n_splits=5,
        fold=1,
        test_ratio=0.2,
        seed=5,
    )

    assert set(split_fold0.test.tolist()) == set(split_fold1.test.tolist())
