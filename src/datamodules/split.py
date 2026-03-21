"""Dataset split helpers used by datamodules and orchestration scripts.

The module intentionally stays framework-agnostic. Every function returns plain
numpy index arrays wrapped in `SplitIndices`, so runtime code can decide how the
indices are consumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np


@dataclass
class SplitIndices:
    """Container for train/val/test sample indices."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    @classmethod
    def from_mapping(cls, payload: Dict[str, Any]) -> "SplitIndices":
        """Create a split container from a plain mapping.

        Args:
            payload: Mapping with optional `train`, `val`, and `test` index lists.

        Returns:
            SplitIndices: Parsed split container with numpy arrays.
        """

        return cls(
            train=np.asarray(payload.get("train", []), dtype=int),
            val=np.asarray(payload.get("val", []), dtype=int),
            test=np.asarray(payload.get("test", []), dtype=int),
        )

    def to_dict(self) -> Dict[str, list[int]]:
        """Convert split indices to plain Python lists.

        Returns:
            Dict[str, list[int]]: Serializable split mapping.
        """

        return {
            "train": self.train.astype(int).tolist(),
            "val": self.val.astype(int).tolist(),
            "test": self.test.astype(int).tolist(),
        }


class SplitProvider:
    """Base interface for framework-level split providers."""

    def build(self) -> SplitIndices:
        """Build split indices for a concrete dataset view.

        Returns:
            SplitIndices: Generated train/validation/test indices.
        """

        raise NotImplementedError


@dataclass
class HoldoutSplitProvider(SplitProvider):
    num_samples: int
    val_ratio: float = 0.2
    seed: int = 42
    candidate_indices: Optional[Sequence[int]] = None

    def build(self) -> SplitIndices:
        """Create holdout split indices from provider settings.

        Returns:
            SplitIndices: Generated holdout split.
        """

        return make_holdout_split(
            num_samples=self.num_samples,
            val_ratio=self.val_ratio,
            seed=self.seed,
            candidate_indices=self.candidate_indices,
        )


@dataclass
class KFoldSplitProvider(SplitProvider):
    num_samples: int
    n_splits: int
    fold: int
    seed: int = 42
    candidate_indices: Optional[Sequence[int]] = None

    def build(self) -> SplitIndices:
        """Create K-fold split indices from provider settings.

        Returns:
            SplitIndices: Generated K-fold split for the configured fold.
        """

        return make_kfold_split(
            num_samples=self.num_samples,
            n_splits=self.n_splits,
            fold=self.fold,
            seed=self.seed,
            candidate_indices=self.candidate_indices,
        )


@dataclass
class GroupHoldoutSplitProvider(SplitProvider):
    num_samples: int
    group_ids: Sequence[int]
    val_ratio: float = 0.2
    seed: int = 42
    candidate_indices: Optional[Sequence[int]] = None

    def build(self) -> SplitIndices:
        """Create group-aware holdout split indices from provider settings.

        Returns:
            SplitIndices: Generated group-aware holdout split.
        """

        return make_group_holdout_split(
            num_samples=self.num_samples,
            group_ids=self.group_ids,
            val_ratio=self.val_ratio,
            seed=self.seed,
            candidate_indices=self.candidate_indices,
        )


@dataclass
class GroupKFoldSplitProvider(SplitProvider):
    num_samples: int
    group_ids: Sequence[int]
    n_splits: int
    fold: int
    seed: int = 42
    candidate_indices: Optional[Sequence[int]] = None

    def build(self) -> SplitIndices:
        """Create group-aware K-fold split indices from provider settings.

        Returns:
            SplitIndices: Generated group-aware K-fold split for the configured fold.
        """

        return make_group_kfold_split(
            num_samples=self.num_samples,
            group_ids=self.group_ids,
            n_splits=self.n_splits,
            fold=self.fold,
            seed=self.seed,
            candidate_indices=self.candidate_indices,
        )


def _as_index_array(values: Optional[Sequence[int]], num_samples: int) -> np.ndarray:
    """Return candidate sample indices as a 1-D integer array.

    Args:
        values: Optional explicit candidate indices.
        num_samples: Total sample count used when `values` is omitted.

    Returns:
        np.ndarray: One-dimensional integer index array.
    """

    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    if values is None:
        return np.arange(num_samples, dtype=int)
    return np.asarray(values, dtype=int)


def _as_group_array(group_ids: Sequence[int], num_samples: int) -> np.ndarray:
    """Return group ids as a numpy array and validate the length.

    Args:
        group_ids: Group identifier for each sample.
        num_samples: Expected number of samples.

    Returns:
        np.ndarray: Group id array aligned with sample indices.
    """

    groups = np.asarray(group_ids)
    if len(groups) != num_samples:
        raise ValueError("group_ids length must match num_samples")
    return groups


def _empty_indices() -> np.ndarray:
    """Return an empty integer index array.

    Returns:
        np.ndarray: Empty integer array used for missing partitions.
    """

    return np.array([], dtype=int)


def _validate_ratio(name: str, value: float) -> None:
    """Validate that a split ratio lies in the open interval `(0, 1)`.

    Args:
        name: Human-readable ratio name for error messages.
        value: Ratio value to validate.

    Returns:
        None: The function raises on invalid values.
    """

    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be in the open interval (0, 1)")


def _validate_kfold_args(n_splits: int, fold: int) -> None:
    """Validate common K-fold arguments.

    Args:
        n_splits: Number of folds.
        fold: Fold index requested by the caller.

    Returns:
        None: The function raises on invalid values.
    """

    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if not 0 <= fold < n_splits:
        raise ValueError(f"fold={fold} is out of range [0, {n_splits - 1}]")


def make_holdout_split(
    num_samples: int,
    val_ratio: float = 0.2,
    seed: int = 42,
    candidate_indices: Optional[Sequence[int]] = None,
) -> SplitIndices:
    """Create a random train/validation holdout split.

    Args:
        num_samples: Total sample count.
        val_ratio: Fraction of candidate samples assigned to validation.
        seed: Random seed used for shuffling.
        candidate_indices: Optional subset of samples that may participate in the split.

    Returns:
        SplitIndices: Holdout split with `train` and `val` populated.
    """

    _validate_ratio("val_ratio", val_ratio)
    indices = _as_index_array(candidate_indices, num_samples).copy()
    if len(indices) < 2:
        raise ValueError("holdout split requires at least 2 candidate samples")

    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    n_val = max(1, int(round(len(indices) * val_ratio)))
    n_val = min(n_val, len(indices) - 1)

    return SplitIndices(
        train=indices[n_val:].astype(int),
        val=indices[:n_val].astype(int),
        test=_empty_indices(),
    )


def make_group_holdout_split(
    num_samples: int,
    group_ids: Sequence[int],
    val_ratio: float = 0.2,
    seed: int = 42,
    candidate_indices: Optional[Sequence[int]] = None,
) -> SplitIndices:
    """Create a group-aware train/validation holdout split.

    Args:
        num_samples: Total sample count.
        group_ids: Group identifier for each sample.
        val_ratio: Target validation ratio measured in samples.
        seed: Random seed used for shuffling groups.
        candidate_indices: Optional subset of samples that may participate in the split.

    Returns:
        SplitIndices: Group-aware holdout split with `train` and `val` populated.
    """

    _validate_ratio("val_ratio", val_ratio)
    indices = _as_index_array(candidate_indices, num_samples)
    groups = _as_group_array(group_ids, num_samples)
    if len(indices) < 2:
        raise ValueError("group holdout split requires at least 2 candidate samples")

    unique_groups = np.unique(groups[indices])
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)

    group_to_indices = {group: indices[groups[indices] == group] for group in unique_groups}
    target_val_size = max(1, int(round(len(indices) * val_ratio)))

    val_groups = []
    val_size = 0
    for group in unique_groups:
        if val_size >= target_val_size and val_groups:
            break
        val_groups.append(group)
        val_size += len(group_to_indices[group])

    val_idx = np.concatenate([group_to_indices[group] for group in val_groups]).astype(int)
    train_idx = np.setdiff1d(indices, val_idx, assume_unique=False).astype(int)
    if len(train_idx) == 0:
        raise ValueError("group holdout split produced an empty train partition")

    return SplitIndices(train=train_idx, val=val_idx, test=_empty_indices())


def _balanced_group_bins(
    candidate_indices: np.ndarray,
    group_ids: np.ndarray,
    n_bins: int,
    seed: int,
) -> list[np.ndarray]:
    """Assign full groups to balanced bins using a greedy heuristic.

    Args:
        candidate_indices: Sample indices eligible for binning.
        group_ids: Group id array aligned with all samples.
        n_bins: Number of bins to build.
        seed: Random seed used before ordering groups greedily.

    Returns:
        list[np.ndarray]: Group-preserving bins of sample indices.
    """

    local_group_ids = group_ids[candidate_indices]
    unique_groups = np.unique(local_group_ids)

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)

    group_to_indices = {
        group: candidate_indices[local_group_ids == group]
        for group in unique_groups
    }
    ordered_groups = sorted(
        unique_groups,
        key=lambda group: len(group_to_indices[group]),
        reverse=True,
    )

    bins: list[list[int]] = [[] for _ in range(n_bins)]
    bin_sizes = [0 for _ in range(n_bins)]

    for group in ordered_groups:
        target_bin = int(np.argmin(bin_sizes))
        bins[target_bin].append(int(group))
        bin_sizes[target_bin] += len(group_to_indices[group])

    output: list[np.ndarray] = []
    for groups_in_bin in bins:
        if not groups_in_bin:
            output.append(_empty_indices())
            continue
        output.append(
            np.concatenate([group_to_indices[group] for group in groups_in_bin]).astype(int)
        )
    return output


def make_kfold_split(
    num_samples: int,
    n_splits: int,
    fold: int,
    seed: int = 42,
    candidate_indices: Optional[Sequence[int]] = None,
) -> SplitIndices:
    """Create one fold of a standard K-fold split.

    Args:
        num_samples: Total sample count.
        n_splits: Number of folds.
        fold: Fold index used as validation.
        seed: Random seed used for shuffling.
        candidate_indices: Optional subset of samples that may participate in the split.

    Returns:
        SplitIndices: K-fold split with `train` and `val` populated.
    """

    _validate_kfold_args(n_splits, fold)
    indices = _as_index_array(candidate_indices, num_samples).copy()
    if len(indices) < n_splits:
        raise ValueError("candidate sample count must be >= n_splits")

    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    fold_parts = np.array_split(indices, n_splits)
    val_idx = fold_parts[fold].astype(int)
    train_idx = np.concatenate(
        [part for idx, part in enumerate(fold_parts) if idx != fold]
    ).astype(int)

    return SplitIndices(train=train_idx, val=val_idx, test=_empty_indices())


def make_group_kfold_split(
    num_samples: int,
    group_ids: Sequence[int],
    n_splits: int,
    fold: int,
    seed: int = 42,
    candidate_indices: Optional[Sequence[int]] = None,
) -> SplitIndices:
    """Create one fold of a group-aware K-fold split.

    Args:
        num_samples: Total sample count.
        group_ids: Group identifier for each sample.
        n_splits: Number of folds.
        fold: Fold index used as validation.
        seed: Random seed used before balancing groups.
        candidate_indices: Optional subset of samples that may participate in the split.

        Returns:
            SplitIndices: Group-aware K-fold split with `train` and `val` populated.
    """

    _validate_kfold_args(n_splits, fold)
    indices = _as_index_array(candidate_indices, num_samples)
    groups = _as_group_array(group_ids, num_samples)
    if len(indices) < n_splits:
        raise ValueError("candidate sample count must be >= n_splits")

    bins = _balanced_group_bins(
        candidate_indices=indices,
        group_ids=groups,
        n_bins=n_splits,
        seed=seed,
    )

    val_idx = bins[fold]
    train_idx = np.concatenate(
        [part for idx, part in enumerate(bins) if idx != fold]
    ).astype(int)
    if len(val_idx) == 0:
        raise ValueError("group K-fold produced an empty validation partition")

    return SplitIndices(train=train_idx, val=val_idx, test=_empty_indices())


def make_dev_test_holdout_split(
    num_samples: int,
    test_ratio: float = 0.2,
    seed: int = 42,
    candidate_indices: Optional[Sequence[int]] = None,
) -> SplitIndices:
    """Create a random dev/test holdout split.

    The dev partition is stored in `train` so callers can apply a second-stage
    train/val split later.

    Args:
        num_samples: Total sample count.
        test_ratio: Fraction of candidate samples assigned to test.
        seed: Random seed used for shuffling.
        candidate_indices: Optional subset of samples that may participate in the split.

    Returns:
        SplitIndices: Split with `train` as development partition and `test` populated.
    """

    _validate_ratio("test_ratio", test_ratio)
    indices = _as_index_array(candidate_indices, num_samples).copy()
    if len(indices) < 2:
        raise ValueError("dev/test split requires at least 2 candidate samples")

    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    n_test = max(1, int(round(len(indices) * test_ratio)))
    n_test = min(n_test, len(indices) - 1)

    return SplitIndices(
        train=indices[n_test:].astype(int),
        val=_empty_indices(),
        test=indices[:n_test].astype(int),
    )


def make_group_dev_test_holdout_split(
    num_samples: int,
    group_ids: Sequence[int],
    test_ratio: float = 0.2,
    seed: int = 42,
    candidate_indices: Optional[Sequence[int]] = None,
) -> SplitIndices:
    """Create a group-aware dev/test holdout split.

    Args:
        num_samples: Total sample count.
        group_ids: Group identifier for each sample.
        test_ratio: Target test ratio measured in samples.
        seed: Random seed used for shuffling groups.
        candidate_indices: Optional subset of samples that may participate in the split.

    Returns:
        SplitIndices: Group-aware split with `train` as development partition.
    """

    _validate_ratio("test_ratio", test_ratio)
    indices = _as_index_array(candidate_indices, num_samples)
    groups = _as_group_array(group_ids, num_samples)
    if len(indices) < 2:
        raise ValueError("group dev/test split requires at least 2 candidate samples")

    unique_groups = np.unique(groups[indices])
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)

    group_to_indices = {group: indices[groups[indices] == group] for group in unique_groups}
    target_test_size = max(1, int(round(len(indices) * test_ratio)))

    test_groups = []
    test_size = 0
    for group in unique_groups:
        if test_size >= target_test_size and test_groups:
            break
        test_groups.append(group)
        test_size += len(group_to_indices[group])

    test_idx = np.concatenate([group_to_indices[group] for group in test_groups]).astype(int)
    dev_idx = np.setdiff1d(indices, test_idx, assume_unique=False).astype(int)
    if len(dev_idx) == 0:
        raise ValueError("group dev/test split produced an empty dev partition")

    return SplitIndices(train=dev_idx, val=_empty_indices(), test=test_idx)
