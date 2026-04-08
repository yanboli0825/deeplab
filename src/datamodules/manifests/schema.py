from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ManifestColumns:
    """Recommended column names for dataset tabular."""

    sample_id: str = "sample_id"
    label: str = "label"
    group: Optional[str] = "group"
    path: Optional[str] = "path"

    def to_dict(self) -> Dict[str, Optional[str]]:
        """Convert the manifest column contract to a plain mapping.

        Returns:
            Dict[str, Optional[str]]: Serializable manifest column definition.
        """

        return {
            "sample_id": self.sample_id,
            "label": self.label,
            "group": self.group,
            "path": self.path,
        }
