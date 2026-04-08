"""Dataset split helpers used by datamodules and orchestration scripts.

The module intentionally stays framework-agnostic. Every function returns plain
numpy index arrays wrapped in `SplitIndices`, so runtime code can decide how the
indices are consumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass
class SplitIndices:
    """Container for train/val/test sample indices."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    @classmethod
    def from_mapping(cls, payload: Dict[str, Any]) -> "SplitIndices":
        """Create a split container from a plain mapping dict.

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
        """Convert split indices to plain Python dict.

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

    def build(self) -> SplitIndices:
        """Create holdout split indices from provider settings.

        Returns:
            SplitIndices: Generated holdout split.
        """

        return make_holdout_split(
            num_samples=self.num_samples,
            val_ratio=self.val_ratio,
            seed=self.seed,
        )


@dataclass
class KFoldSplitProvider(SplitProvider):
    num_samples: int
    n_splits: int
    fold: int
    seed: int = 42

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
        )


@dataclass
class StratifiedHoldoutSplitProvider(SplitProvider):
    num_samples: int
    labels: Sequence[Any]
    val_ratio: float = 0.2
    seed: int = 42

    def build(self) -> SplitIndices:
        """Create stratified holdout split indices from provider settings."""

        return make_stratified_holdout_split(
            num_samples=self.num_samples,
            labels=self.labels,
            val_ratio=self.val_ratio,
            seed=self.seed,
        )


@dataclass
class StratifiedKFoldSplitProvider(SplitProvider):
    num_samples: int
    labels: Sequence[Any]
    n_splits: int
    fold: int
    seed: int = 42

    def build(self) -> SplitIndices:
        """Create stratified K-fold split indices from provider settings."""

        return make_stratified_kfold_split(
            num_samples=self.num_samples,
            labels=self.labels,
            n_splits=self.n_splits,
            fold=self.fold,
            seed=self.seed,
        )


@dataclass
class GroupHoldoutSplitProvider(SplitProvider):
    num_samples: int
    group_ids: Sequence[Any]
    val_ratio: float = 0.2
    seed: int = 42

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
        )


@dataclass
class GroupKFoldSplitProvider(SplitProvider):
    num_samples: int
    group_ids: Sequence[Any]
    n_splits: int
    fold: int
    seed: int = 42

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
        )


@dataclass
class StratifiedGroupHoldoutSplitProvider(SplitProvider):
    num_samples: int
    group_ids: Sequence[Any]
    labels: Sequence[Any]
    val_ratio: float = 0.2
    seed: int = 42

    def build(self) -> SplitIndices:
        """Create a stratified group-aware holdout split."""

        return make_stratified_group_holdout_split(
            num_samples=self.num_samples,
            group_ids=self.group_ids,
            labels=self.labels,
            val_ratio=self.val_ratio,
            seed=self.seed,
        )


@dataclass
class StratifiedGroupKFoldSplitProvider(SplitProvider):
    num_samples: int
    group_ids: Sequence[Any]
    labels: Sequence[Any]
    n_splits: int
    fold: int
    seed: int = 42

    def build(self) -> SplitIndices:
        """Create a stratified group-aware K-fold split."""

        return make_stratified_group_kfold_split(
            num_samples=self.num_samples,
            group_ids=self.group_ids,
            labels=self.labels,
            n_splits=self.n_splits,
            fold=self.fold,
            seed=self.seed,
        )


@dataclass
class TrainValTestHoldoutSplitProvider(SplitProvider):
    num_samples: int
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    seed: int = 42

    def build(self) -> SplitIndices:
        """Create a train/validation/test holdout split from provider settings.

        Returns:
            SplitIndices: Generated three-way holdout split.
        """

        return make_train_val_test_holdout_split(
            num_samples=self.num_samples,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            seed=self.seed,
        )


@dataclass
class StratifiedTrainValTestHoldoutSplitProvider(SplitProvider):
    num_samples: int
    labels: Sequence[Any]
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    seed: int = 42

    def build(self) -> SplitIndices:
        """Create a stratified train/validation/test holdout split."""

        return make_stratified_train_val_test_holdout_split(
            num_samples=self.num_samples,
            labels=self.labels,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            seed=self.seed,
        )


@dataclass
class GroupTrainValTestHoldoutSplitProvider(SplitProvider):
    num_samples: int
    group_ids: Sequence[Any]
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    seed: int = 42

    def build(self) -> SplitIndices:
        """Create a group-aware train/validation/test holdout split.

        Returns:
            SplitIndices: Generated group-aware three-way holdout split.
        """

        return make_group_train_val_test_holdout_split(
            num_samples=self.num_samples,
            group_ids=self.group_ids,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            seed=self.seed,
        )


@dataclass
class StratifiedGroupTrainValTestHoldoutSplitProvider(SplitProvider):
    num_samples: int
    group_ids: Sequence[Any]
    labels: Sequence[Any]
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    seed: int = 42

    def build(self) -> SplitIndices:
        """Create a stratified group-aware train/validation/test holdout split."""

        return make_stratified_group_train_val_test_holdout_split(
            num_samples=self.num_samples,
            group_ids=self.group_ids,
            labels=self.labels,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            seed=self.seed,
        )


@dataclass
class DevTestKFoldSplitProvider(SplitProvider):
    num_samples: int
    n_splits: int
    fold: int
    test_ratio: float = 0.2
    seed: int = 42

    def build(self) -> SplitIndices:
        """Create a fixed-test K-fold split from provider settings.

        Returns:
            SplitIndices: Generated split with fixed `test` and fold-specific `train/val`.
        """

        return make_dev_test_kfold_split(
            num_samples=self.num_samples,
            n_splits=self.n_splits,
            fold=self.fold,
            test_ratio=self.test_ratio,
            seed=self.seed,
        )


@dataclass
class StratifiedDevTestKFoldSplitProvider(SplitProvider):
    num_samples: int
    labels: Sequence[Any]
    n_splits: int
    fold: int
    test_ratio: float = 0.2
    seed: int = 42

    def build(self) -> SplitIndices:
        """Create a stratified fixed-test K-fold split."""

        return make_stratified_dev_test_kfold_split(
            num_samples=self.num_samples,
            labels=self.labels,
            n_splits=self.n_splits,
            fold=self.fold,
            test_ratio=self.test_ratio,
            seed=self.seed,
        )


@dataclass
class GroupDevTestKFoldSplitProvider(SplitProvider):
    num_samples: int
    group_ids: Sequence[Any]
    n_splits: int
    fold: int
    test_ratio: float = 0.2
    seed: int = 42

    def build(self) -> SplitIndices:
        """Create a group-aware fixed-test K-fold split from provider settings.

        Returns:
            SplitIndices: Generated group-aware split with fixed `test` and fold-specific `train/val`.
        """

        return make_group_dev_test_kfold_split(
            num_samples=self.num_samples,
            group_ids=self.group_ids,
            n_splits=self.n_splits,
            fold=self.fold,
            test_ratio=self.test_ratio,
            seed=self.seed,
        )


@dataclass
class StratifiedGroupDevTestKFoldSplitProvider(SplitProvider):
    num_samples: int
    group_ids: Sequence[Any]
    labels: Sequence[Any]
    n_splits: int
    fold: int
    test_ratio: float = 0.2
    seed: int = 42

    def build(self) -> SplitIndices:
        """Create a stratified group-aware fixed-test K-fold split."""

        return make_stratified_group_dev_test_kfold_split(
            num_samples=self.num_samples,
            group_ids=self.group_ids,
            labels=self.labels,
            n_splits=self.n_splits,
            fold=self.fold,
            test_ratio=self.test_ratio,
            seed=self.seed,
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


def _as_group_array(group_ids: Sequence[Any], num_samples: int) -> np.ndarray:
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


def _as_label_array(labels: Sequence[Any], num_samples: int) -> np.ndarray:
    """Return labels as a numpy array and validate the length."""

    values = np.asarray(labels)
    if len(values) != num_samples:
        raise ValueError("labels length must match num_samples")
    return values


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


def _require_sklearn() -> Tuple[Any, Any]:
    """Import stratified split helpers lazily."""

    try:
        from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
    except ImportError as exc:
        raise ImportError(
            "Stratified split methods require scikit-learn to be installed."
        ) from exc
    return StratifiedKFold, StratifiedShuffleSplit


def _ordered_group_records(
    candidate_indices: np.ndarray,
    group_ids: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> Tuple[list[dict[str, Any]], np.ndarray]:
    """Build per-group label statistics used by stratified group split heuristics."""

    local_groups = group_ids[candidate_indices]
    local_labels = labels[candidate_indices]
    classes = np.unique(local_labels)
    class_to_idx = {value: idx for idx, value in enumerate(classes)}
    unique_groups = np.unique(local_groups)

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)

    records: list[dict[str, Any]] = []
    for group in unique_groups:
        indices = candidate_indices[local_groups == group]
        group_labels = labels[indices]
        counts = np.zeros(len(classes), dtype=float)
        for label in group_labels:
            counts[class_to_idx[label]] += 1.0
        dominant_ratio = float(counts.max() / counts.sum()) if counts.sum() > 0 else 0.0
        records.append(
            {
                "group": group,
                "indices": indices.astype(int),
                "counts": counts,
                "size": int(len(indices)),
                "dominant_ratio": dominant_ratio,
            }
        )

    records.sort(
        key=lambda item: (item["size"], item["dominant_ratio"]),
        reverse=True,
    )
    return records, classes


def _partition_score(
    size: float,
    counts: np.ndarray,
    target_size: float,
    target_counts: np.ndarray,
) -> float:
    """Score how close a partition is to the desired sample/label distribution."""

    size_denominator = max(1.0, target_size)
    count_denominator = np.maximum(1.0, target_counts)
    size_penalty = abs(size - target_size) / size_denominator
    count_penalty = float(np.sum(np.abs(counts - target_counts) / count_denominator))
    return size_penalty + count_penalty


def _partition_cost_tuple(
    size: float,
    counts: np.ndarray,
    target_size: float,
    target_counts: np.ndarray,
) -> tuple[float, float]:
    """Return lexicographic `(label_cost, size_cost)` for a candidate partition."""

    count_denominator = np.maximum(1.0, target_counts)
    label_cost = float(np.sum(np.abs(counts - target_counts) / count_denominator))
    size_cost = abs(size - target_size) / max(1.0, target_size)
    return (label_cost, size_cost)


def _total_cost_tuple(
    bin_sizes: Sequence[float],
    bin_counts: Sequence[np.ndarray],
    target_sizes: np.ndarray,
    target_counts: np.ndarray,
) -> tuple[float, float]:
    """Aggregate lexicographic cost across all bins."""

    label_total = 0.0
    size_total = 0.0
    for idx in range(len(bin_sizes)):
        label_cost, size_cost = _partition_cost_tuple(
            size=bin_sizes[idx],
            counts=bin_counts[idx],
            target_size=float(target_sizes[idx]),
            target_counts=target_counts[idx],
        )
        label_total += label_cost
        size_total += size_cost
    return (label_total, size_total)


def _unique_group_count(candidate_indices: np.ndarray, group_ids: np.ndarray) -> int:
    """Count unique groups inside a candidate subset."""

    if len(candidate_indices) == 0:
        return 0
    return int(len(np.unique(group_ids[candidate_indices])))


def _class_group_coverage(
    candidate_indices: np.ndarray,
    group_ids: np.ndarray,
    labels: np.ndarray,
) -> Dict[Any, int]:
    """Count how many unique groups cover each class in a candidate subset."""

    coverage: Dict[Any, set[Any]] = {}
    for index in candidate_indices.astype(int).tolist():
        label = labels[index]
        group = group_ids[index]
        coverage.setdefault(label, set()).add(group)
    return {label: len(groups) for label, groups in coverage.items()}


def _validate_group_class_coverage(
    candidate_indices: np.ndarray,
    group_ids: np.ndarray,
    labels: np.ndarray,
    required_bins: int,
    context: str,
) -> None:
    """Ensure each class is represented by enough groups for stratified group splitting."""

    coverage = _class_group_coverage(candidate_indices, group_ids, labels)
    for label, count in coverage.items():
        if count < required_bins:
            raise ValueError(
                f"{context} requires each class to appear in at least {required_bins} unique groups; "
                f"class '{label}' only appears in {count} group(s)."
            )


def _counts_cover_all_classes(counts: np.ndarray) -> bool:
    """Return whether a count vector contains every class at least once."""

    return bool(np.all(counts > 0))


def _stratified_group_holdout_partition(
    candidate_indices: np.ndarray,
    group_ids: np.ndarray,
    labels: np.ndarray,
    ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split candidate samples into a stratified group-aware target/rest partition."""

    bins = _stratified_group_bins(
        candidate_indices=candidate_indices,
        group_ids=group_ids,
        labels=labels,
        n_bins=2,
        seed=seed,
        target_proportions=np.asarray([ratio, 1.0 - ratio], dtype=float),
    )
    target_idx = bins[0]
    remainder_idx = bins[1]
    if len(target_idx) == 0 or len(remainder_idx) == 0:
        raise ValueError("stratified group holdout produced an empty partition")
    return target_idx, remainder_idx


def _stratified_group_bins(
    candidate_indices: np.ndarray,
    group_ids: np.ndarray,
    labels: np.ndarray,
    n_bins: int,
    seed: int,
    target_proportions: Optional[np.ndarray] = None,
) -> list[np.ndarray]:
    """Assign groups to bins while prioritizing label balance and keeping bins non-empty."""

    records, _ = _ordered_group_records(candidate_indices, group_ids, labels, seed)
    if len(records) < n_bins:
        raise ValueError("stratified group K-fold requires at least n_splits unique groups")

    total_size = float(len(candidate_indices))
    total_counts = np.zeros_like(records[0]["counts"]) if records else np.array([], dtype=float)
    for record in records:
        total_counts = total_counts + record["counts"]

    if target_proportions is None:
        target_proportions = np.full(n_bins, 1.0 / float(n_bins), dtype=float)
    target_sizes = target_proportions * total_size
    target_counts = np.vstack([total_counts * proportion for proportion in target_proportions])
    bin_sizes = [0.0 for _ in range(n_bins)]
    bin_counts = [np.zeros_like(total_counts) for _ in range(n_bins)]
    bin_records: list[list[dict[str, Any]]] = [[] for _ in range(n_bins)]

    remaining = list(records)
    missing_classes = [set(np.where(total_counts > 0)[0].tolist()) for _ in range(n_bins)]

    # Phase 1: ensure every bin becomes non-empty and covers as many missing classes as possible.
    for bin_idx in range(n_bins):
        best_pos = None
        best_key: tuple[int, int, float, float] | None = None
        for pos, record in enumerate(remaining):
            covered_now = int(np.count_nonzero(record["counts"] > 0))
            missing_after = int(
                len(missing_classes[bin_idx] - set(np.where(record["counts"] > 0)[0].tolist()))
            )
            label_cost, size_cost = _partition_cost_tuple(
                size=record["size"],
                counts=record["counts"],
                target_size=float(target_sizes[bin_idx]),
                target_counts=target_counts[bin_idx],
            )
            key = (missing_after, -covered_now, label_cost, size_cost)
            if best_key is None or key < best_key:
                best_key = key
                best_pos = pos
        if best_pos is None:
            break
        record = remaining.pop(best_pos)
        bin_records[bin_idx].append(record)
        bin_sizes[bin_idx] += record["size"]
        bin_counts[bin_idx] = bin_counts[bin_idx] + record["counts"]
        missing_classes[bin_idx] -= set(np.where(record["counts"] > 0)[0].tolist())

    # Phase 2: prioritize filling missing classes in bins before optimizing distribution.
    while remaining:
        if all(len(missing) == 0 for missing in missing_classes):
            break
        best_choice: tuple[int, int] | None = None
        best_key: tuple[int, float, float, float] | None = None
        for pos, record in enumerate(remaining):
            record_classes = set(np.where(record["counts"] > 0)[0].tolist())
            for bin_idx in range(n_bins):
                missing_before = len(missing_classes[bin_idx])
                if missing_before == 0:
                    continue
                missing_after_set = missing_classes[bin_idx] - record_classes
                missing_after = len(missing_after_set)
                classes_filled = missing_before - missing_after
                if classes_filled <= 0:
                    continue
                label_cost, size_cost = _partition_cost_tuple(
                    size=bin_sizes[bin_idx] + record["size"],
                    counts=bin_counts[bin_idx] + record["counts"],
                    target_size=float(target_sizes[bin_idx]),
                    target_counts=target_counts[bin_idx],
                )
                key = (-classes_filled, label_cost, size_cost, bin_sizes[bin_idx])
                if best_key is None or key < best_key:
                    best_key = key
                    best_choice = (pos, bin_idx)
        if best_choice is None:
            break
        record_pos, bin_idx = best_choice
        record = remaining.pop(record_pos)
        bin_records[bin_idx].append(record)
        bin_sizes[bin_idx] += record["size"]
        bin_counts[bin_idx] = bin_counts[bin_idx] + record["counts"]
        missing_classes[bin_idx] -= set(np.where(record["counts"] > 0)[0].tolist())

    # Phase 3: assign the rest by lexicographic (label_cost, size_cost).
    for record in remaining:
        best_bin = 0
        best_cost = (float("inf"), float("inf"))
        for idx in range(n_bins):
            cost = _partition_cost_tuple(
                size=bin_sizes[idx] + record["size"],
                counts=bin_counts[idx] + record["counts"],
                target_size=float(target_sizes[idx]),
                target_counts=target_counts[idx],
            )
            if cost < best_cost:
                best_cost = cost
                best_bin = idx
        bin_records[best_bin].append(record)
        bin_sizes[best_bin] += record["size"]
        bin_counts[best_bin] = bin_counts[best_bin] + record["counts"]

    def current_total_cost() -> tuple[float, float]:
        return _total_cost_tuple(
            bin_sizes=bin_sizes,
            bin_counts=bin_counts,
            target_sizes=target_sizes,
            target_counts=target_counts,
        )

    for _ in range(3):
        improved = False
        baseline_cost = current_total_cost()
        for src_idx in range(n_bins):
            for dst_idx in range(n_bins):
                if src_idx == dst_idx:
                    continue
                for record in list(bin_records[src_idx]):
                    if len(bin_records[src_idx]) <= 1:
                        break
                    candidate_sizes = list(bin_sizes)
                    candidate_counts = [counts.copy() for counts in bin_counts]
                    candidate_records = [items.copy() for items in bin_records]
                    candidate_records[src_idx].remove(record)
                    candidate_records[dst_idx].append(record)
                    candidate_sizes[src_idx] -= record["size"]
                    candidate_sizes[dst_idx] += record["size"]
                    candidate_counts[src_idx] = candidate_counts[src_idx] - record["counts"]
                    candidate_counts[dst_idx] = candidate_counts[dst_idx] + record["counts"]
                    if not all(_counts_cover_all_classes(counts) for counts in candidate_counts):
                        continue
                    candidate_cost = _total_cost_tuple(
                        bin_sizes=candidate_sizes,
                        bin_counts=candidate_counts,
                        target_sizes=target_sizes,
                        target_counts=target_counts,
                    )
                    if candidate_cost < baseline_cost:
                        bin_records = candidate_records
                        bin_sizes = candidate_sizes
                        bin_counts = candidate_counts
                        baseline_cost = candidate_cost
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

        for left_idx in range(n_bins):
            for right_idx in range(left_idx + 1, n_bins):
                for left_record in list(bin_records[left_idx]):
                    for right_record in list(bin_records[right_idx]):
                        candidate_sizes = list(bin_sizes)
                        candidate_counts = [counts.copy() for counts in bin_counts]
                        candidate_records = [items.copy() for items in bin_records]
                        candidate_records[left_idx].remove(left_record)
                        candidate_records[right_idx].remove(right_record)
                        candidate_records[left_idx].append(right_record)
                        candidate_records[right_idx].append(left_record)
                        candidate_sizes[left_idx] += right_record["size"] - left_record["size"]
                        candidate_sizes[right_idx] += left_record["size"] - right_record["size"]
                        candidate_counts[left_idx] = (
                            candidate_counts[left_idx] - left_record["counts"] + right_record["counts"]
                        )
                        candidate_counts[right_idx] = (
                            candidate_counts[right_idx] - right_record["counts"] + left_record["counts"]
                        )
                        if not all(_counts_cover_all_classes(counts) for counts in candidate_counts):
                            continue
                        candidate_cost = _total_cost_tuple(
                            bin_sizes=candidate_sizes,
                            bin_counts=candidate_counts,
                            target_sizes=target_sizes,
                            target_counts=target_counts,
                        )
                        if candidate_cost < baseline_cost:
                            bin_records = candidate_records
                            bin_sizes = candidate_sizes
                            bin_counts = candidate_counts
                            baseline_cost = candidate_cost
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break

    if not all(_counts_cover_all_classes(counts) for counts in bin_counts):
        raise ValueError(
            "stratified group split could not allocate all classes to every partition under group constraints"
        )

    output: list[np.ndarray] = []
    for records_in_bin in bin_records:
        if not records_in_bin:
            output.append(_empty_indices())
            continue
        output.append(
            np.concatenate([record["indices"] for record in records_in_bin]).astype(int)
        )
    return output


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


def make_stratified_holdout_split(
    num_samples: int,
    labels: Sequence[Any],
    val_ratio: float = 0.2,
    seed: int = 42,
    candidate_indices: Optional[Sequence[int]] = None,
) -> SplitIndices:
    """Create a label-stratified train/validation holdout split."""

    _validate_ratio("val_ratio", val_ratio)
    indices = _as_index_array(candidate_indices, num_samples).copy()
    values = _as_label_array(labels, num_samples)
    if len(indices) < 2:
        raise ValueError("stratified holdout split requires at least 2 candidate samples")

    _, StratifiedShuffleSplit = _require_sklearn()
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    try:
        train_pos, val_pos = next(splitter.split(indices, values[indices]))
    except ValueError as exc:
        raise ValueError(f"Unable to build stratified holdout split: {exc}") from exc

    return SplitIndices(
        train=indices[train_pos].astype(int),
        val=indices[val_pos].astype(int),
        test=_empty_indices(),
    )


def make_group_holdout_split(
    num_samples: int,
    group_ids: Sequence[Any],
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


def make_stratified_group_holdout_split(
    num_samples: int,
    group_ids: Sequence[Any],
    labels: Sequence[Any],
    val_ratio: float = 0.2,
    seed: int = 42,
    candidate_indices: Optional[Sequence[int]] = None,
) -> SplitIndices:
    """Create a stratified group-aware train/validation holdout split."""

    _validate_ratio("val_ratio", val_ratio)
    indices = _as_index_array(candidate_indices, num_samples)
    groups = _as_group_array(group_ids, num_samples)
    values = _as_label_array(labels, num_samples)
    if len(indices) < 2:
        raise ValueError("stratified group holdout split requires at least 2 candidate samples")
    _validate_group_class_coverage(
        candidate_indices=indices,
        group_ids=groups,
        labels=values,
        required_bins=2,
        context="stratified group holdout",
    )

    val_idx, train_idx = _stratified_group_holdout_partition(
        candidate_indices=indices,
        group_ids=groups,
        labels=values,
        ratio=val_ratio,
        seed=seed,
    )
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

    bins: list[list[Any]] = [[] for _ in range(n_bins)]
    bin_sizes = [0 for _ in range(n_bins)]

    for group in ordered_groups:
        target_bin = int(np.argmin(bin_sizes))
        bins[target_bin].append(group)
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


def make_stratified_kfold_split(
    num_samples: int,
    labels: Sequence[Any],
    n_splits: int,
    fold: int,
    seed: int = 42,
    candidate_indices: Optional[Sequence[int]] = None,
) -> SplitIndices:
    """Create one fold of a label-stratified K-fold split."""

    _validate_kfold_args(n_splits, fold)
    indices = _as_index_array(candidate_indices, num_samples).copy()
    values = _as_label_array(labels, num_samples)
    if len(indices) < n_splits:
        raise ValueError("candidate sample count must be >= n_splits")

    StratifiedKFold, _ = _require_sklearn()
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    try:
        folds = list(splitter.split(indices, values[indices]))
    except ValueError as exc:
        raise ValueError(f"Unable to build stratified K-fold split: {exc}") from exc
    train_pos, val_pos = folds[fold]

    return SplitIndices(
        train=indices[train_pos].astype(int),
        val=indices[val_pos].astype(int),
        test=_empty_indices(),
    )


def make_group_kfold_split(
    num_samples: int,
    group_ids: Sequence[Any],
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
    unique_group_count = _unique_group_count(indices, groups)
    if unique_group_count < n_splits:
        raise ValueError("group-aware K-fold requires at least n_splits unique groups")

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


def make_stratified_group_kfold_split(
    num_samples: int,
    group_ids: Sequence[Any],
    labels: Sequence[Any],
    n_splits: int,
    fold: int,
    seed: int = 42,
    candidate_indices: Optional[Sequence[int]] = None,
) -> SplitIndices:
    """Create one fold of a stratified group-aware K-fold split."""

    _validate_kfold_args(n_splits, fold)
    indices = _as_index_array(candidate_indices, num_samples)
    groups = _as_group_array(group_ids, num_samples)
    values = _as_label_array(labels, num_samples)
    if len(indices) < n_splits:
        raise ValueError("candidate sample count must be >= n_splits")
    unique_group_count = _unique_group_count(indices, groups)
    if unique_group_count < n_splits:
        raise ValueError("stratified group K-fold requires at least n_splits unique groups")
    _validate_group_class_coverage(
        candidate_indices=indices,
        group_ids=groups,
        labels=values,
        required_bins=n_splits,
        context="stratified group K-fold",
    )

    bins = _stratified_group_bins(
        candidate_indices=indices,
        group_ids=groups,
        labels=values,
        n_bins=n_splits,
        seed=seed,
    )
    val_idx = bins[fold]
    if len(val_idx) == 0:
        raise ValueError("stratified group K-fold produced an empty validation partition")
    train_idx = np.concatenate(
        [part for idx, part in enumerate(bins) if idx != fold]
    ).astype(int)
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


def make_stratified_dev_test_holdout_split(
    num_samples: int,
    labels: Sequence[Any],
    test_ratio: float = 0.2,
    seed: int = 42,
    candidate_indices: Optional[Sequence[int]] = None,
) -> SplitIndices:
    """Create a label-stratified dev/test holdout split."""

    _validate_ratio("test_ratio", test_ratio)
    indices = _as_index_array(candidate_indices, num_samples).copy()
    values = _as_label_array(labels, num_samples)
    if len(indices) < 2:
        raise ValueError("stratified dev/test split requires at least 2 candidate samples")

    _, StratifiedShuffleSplit = _require_sklearn()
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    try:
        dev_pos, test_pos = next(splitter.split(indices, values[indices]))
    except ValueError as exc:
        raise ValueError(f"Unable to build stratified dev/test split: {exc}") from exc

    return SplitIndices(
        train=indices[dev_pos].astype(int),
        val=_empty_indices(),
        test=indices[test_pos].astype(int),
    )


def make_group_dev_test_holdout_split(
    num_samples: int,
    group_ids: Sequence[Any],
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


def make_stratified_group_dev_test_holdout_split(
    num_samples: int,
    group_ids: Sequence[Any],
    labels: Sequence[Any],
    test_ratio: float = 0.2,
    seed: int = 42,
    candidate_indices: Optional[Sequence[int]] = None,
) -> SplitIndices:
    """Create a stratified group-aware dev/test holdout split."""

    _validate_ratio("test_ratio", test_ratio)
    indices = _as_index_array(candidate_indices, num_samples)
    groups = _as_group_array(group_ids, num_samples)
    values = _as_label_array(labels, num_samples)
    if len(indices) < 2:
        raise ValueError("stratified group dev/test split requires at least 2 candidate samples")
    _validate_group_class_coverage(
        candidate_indices=indices,
        group_ids=groups,
        labels=values,
        required_bins=2,
        context="stratified group dev/test holdout",
    )

    test_idx, dev_idx = _stratified_group_holdout_partition(
        candidate_indices=indices,
        group_ids=groups,
        labels=values,
        ratio=test_ratio,
        seed=seed,
    )
    return SplitIndices(train=dev_idx, val=_empty_indices(), test=test_idx)


def make_train_val_test_holdout_split(
    num_samples: int,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
    candidate_indices: Optional[Sequence[int]] = None,
) -> SplitIndices:
    """Create a two-stage train/validation/test holdout split.

    The split is produced in two stages:
    1. hold out `test` from the candidate pool
    2. split the remaining development pool into `train` and `val`

    Args:
        num_samples: Total sample count.
        val_ratio: Validation ratio measured on the development partition.
        test_ratio: Test ratio measured on the candidate partition.
        seed: Random seed used for deterministic shuffling.
        candidate_indices: Optional subset of samples that may participate in the split.

    Returns:
        SplitIndices: Three-way split with populated `train`, `val`, and `test`.
    """

    dev_test = make_dev_test_holdout_split(
        num_samples=num_samples,
        test_ratio=test_ratio,
        seed=seed,
        candidate_indices=candidate_indices,
    )
    train_val = make_holdout_split(
        num_samples=num_samples,
        val_ratio=val_ratio,
        seed=seed,
        candidate_indices=dev_test.train,
    )
    return SplitIndices(
        train=train_val.train,
        val=train_val.val,
        test=dev_test.test,
    )


def make_stratified_train_val_test_holdout_split(
    num_samples: int,
    labels: Sequence[Any],
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
    candidate_indices: Optional[Sequence[int]] = None,
) -> SplitIndices:
    """Create a label-stratified train/validation/test holdout split."""

    values = _as_label_array(labels, num_samples)
    dev_test = make_stratified_dev_test_holdout_split(
        num_samples=num_samples,
        labels=values,
        test_ratio=test_ratio,
        seed=seed,
        candidate_indices=candidate_indices,
    )
    train_val = make_stratified_holdout_split(
        num_samples=num_samples,
        labels=values,
        val_ratio=val_ratio,
        seed=seed,
        candidate_indices=dev_test.train,
    )
    return SplitIndices(
        train=train_val.train,
        val=train_val.val,
        test=dev_test.test,
    )


def make_group_train_val_test_holdout_split(
    num_samples: int,
    group_ids: Sequence[Any],
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
    candidate_indices: Optional[Sequence[int]] = None,
) -> SplitIndices:
    """Create a group-aware two-stage train/validation/test holdout split.

    Args:
        num_samples: Total sample count.
        group_ids: Group identifier for each sample.
        val_ratio: Validation ratio measured on the development partition.
        test_ratio: Test ratio measured on the candidate partition.
        seed: Random seed used for deterministic shuffling.
        candidate_indices: Optional subset of samples that may participate in the split.

    Returns:
        SplitIndices: Group-aware three-way split.
    """

    dev_test = make_group_dev_test_holdout_split(
        num_samples=num_samples,
        group_ids=group_ids,
        test_ratio=test_ratio,
        seed=seed,
        candidate_indices=candidate_indices,
    )
    train_val = make_group_holdout_split(
        num_samples=num_samples,
        group_ids=group_ids,
        val_ratio=val_ratio,
        seed=seed,
        candidate_indices=dev_test.train,
    )
    return SplitIndices(
        train=train_val.train,
        val=train_val.val,
        test=dev_test.test,
    )


def make_stratified_group_train_val_test_holdout_split(
    num_samples: int,
    group_ids: Sequence[Any],
    labels: Sequence[Any],
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
    candidate_indices: Optional[Sequence[int]] = None,
) -> SplitIndices:
    """Create a stratified group-aware train/validation/test holdout split."""

    groups = _as_group_array(group_ids, num_samples)
    values = _as_label_array(labels, num_samples)
    dev_test = make_stratified_group_dev_test_holdout_split(
        num_samples=num_samples,
        group_ids=groups,
        labels=values,
        test_ratio=test_ratio,
        seed=seed,
        candidate_indices=candidate_indices,
    )
    train_val = make_stratified_group_holdout_split(
        num_samples=num_samples,
        group_ids=groups,
        labels=values,
        val_ratio=val_ratio,
        seed=seed,
        candidate_indices=dev_test.train,
    )
    return SplitIndices(
        train=train_val.train,
        val=train_val.val,
        test=dev_test.test,
    )


def make_dev_test_kfold_split(
    num_samples: int,
    n_splits: int,
    fold: int,
    test_ratio: float = 0.2,
    seed: int = 42,
    candidate_indices: Optional[Sequence[int]] = None,
) -> SplitIndices:
    """Create a fixed-test K-fold split.

    The function first holds out a fixed test partition and then runs K-fold on
    the remaining development partition.

    Args:
        num_samples: Total sample count.
        n_splits: Number of folds for the development partition.
        fold: Fold index used as validation.
        test_ratio: Test ratio measured on the candidate partition.
        seed: Random seed used for deterministic shuffling.
        candidate_indices: Optional subset of samples that may participate in the split.

    Returns:
        SplitIndices: Split with fixed `test` and fold-specific `train/val`.
    """

    dev_test = make_dev_test_holdout_split(
        num_samples=num_samples,
        test_ratio=test_ratio,
        seed=seed,
        candidate_indices=candidate_indices,
    )
    train_val = make_kfold_split(
        num_samples=num_samples,
        n_splits=n_splits,
        fold=fold,
        seed=seed,
        candidate_indices=dev_test.train,
    )
    return SplitIndices(
        train=train_val.train,
        val=train_val.val,
        test=dev_test.test,
    )


def make_stratified_dev_test_kfold_split(
    num_samples: int,
    labels: Sequence[Any],
    n_splits: int,
    fold: int,
    test_ratio: float = 0.2,
    seed: int = 42,
    candidate_indices: Optional[Sequence[int]] = None,
) -> SplitIndices:
    """Create a label-stratified fixed-test K-fold split."""

    values = _as_label_array(labels, num_samples)
    dev_test = make_stratified_dev_test_holdout_split(
        num_samples=num_samples,
        labels=values,
        test_ratio=test_ratio,
        seed=seed,
        candidate_indices=candidate_indices,
    )
    train_val = make_stratified_kfold_split(
        num_samples=num_samples,
        labels=values,
        n_splits=n_splits,
        fold=fold,
        seed=seed,
        candidate_indices=dev_test.train,
    )
    return SplitIndices(
        train=train_val.train,
        val=train_val.val,
        test=dev_test.test,
    )


def make_group_dev_test_kfold_split(
    num_samples: int,
    group_ids: Sequence[Any],
    n_splits: int,
    fold: int,
    test_ratio: float = 0.2,
    seed: int = 42,
    candidate_indices: Optional[Sequence[int]] = None,
) -> SplitIndices:
    """Create a group-aware fixed-test K-fold split.

    Args:
        num_samples: Total sample count.
        group_ids: Group identifier for each sample.
        n_splits: Number of folds for the development partition.
        fold: Fold index used as validation.
        test_ratio: Test ratio measured on the candidate partition.
        seed: Random seed used for deterministic shuffling.
        candidate_indices: Optional subset of samples that may participate in the split.

    Returns:
        SplitIndices: Group-aware split with fixed `test` and fold-specific `train/val`.
    """

    dev_test = make_group_dev_test_holdout_split(
        num_samples=num_samples,
        group_ids=group_ids,
        test_ratio=test_ratio,
        seed=seed,
        candidate_indices=candidate_indices,
    )
    dev_groups = _as_group_array(group_ids, num_samples)
    dev_unique_groups = _unique_group_count(dev_test.train, dev_groups)
    if dev_unique_groups < n_splits:
        raise ValueError(
            f"group dev/test K-fold requires at least {n_splits} unique dev groups after test holdout; "
            f"got {dev_unique_groups}. Reduce split.test_ratio or split.n_folds."
        )
    train_val = make_group_kfold_split(
        num_samples=num_samples,
        group_ids=group_ids,
        n_splits=n_splits,
        fold=fold,
        seed=seed,
        candidate_indices=dev_test.train,
    )
    return SplitIndices(
        train=train_val.train,
        val=train_val.val,
        test=dev_test.test,
    )


def make_stratified_group_dev_test_kfold_split(
    num_samples: int,
    group_ids: Sequence[Any],
    labels: Sequence[Any],
    n_splits: int,
    fold: int,
    test_ratio: float = 0.2,
    seed: int = 42,
    candidate_indices: Optional[Sequence[int]] = None,
) -> SplitIndices:
    """Create a stratified group-aware fixed-test K-fold split."""

    groups = _as_group_array(group_ids, num_samples)
    values = _as_label_array(labels, num_samples)
    dev_test = make_stratified_group_dev_test_holdout_split(
        num_samples=num_samples,
        group_ids=groups,
        labels=values,
        test_ratio=test_ratio,
        seed=seed,
        candidate_indices=candidate_indices,
    )
    dev_unique_groups = _unique_group_count(dev_test.train, groups)
    if dev_unique_groups < n_splits:
        raise ValueError(
            f"stratified group dev/test K-fold requires at least {n_splits} unique dev groups after test holdout; "
            f"got {dev_unique_groups}. Reduce split.test_ratio or split.n_folds."
        )
    _validate_group_class_coverage(
        candidate_indices=dev_test.train,
        group_ids=groups,
        labels=values,
        required_bins=n_splits,
        context="stratified group dev/test K-fold (dev partition)",
    )
    train_val = make_stratified_group_kfold_split(
        num_samples=num_samples,
        group_ids=groups,
        labels=values,
        n_splits=n_splits,
        fold=fold,
        seed=seed,
        candidate_indices=dev_test.train,
    )
    return SplitIndices(
        train=train_val.train,
        val=train_val.val,
        test=dev_test.test,
    )
