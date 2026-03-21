from __future__ import annotations

from typing import Iterable, List


def identity_features(values: Iterable[float]) -> List[float]:
    """Return tabular features unchanged after numeric casting.

    Args:
        values: Raw feature values from the manifest or upstream preprocessing.

    Returns:
        List[float]: Feature values converted to Python floats.
    """

    return [float(value) for value in values]
