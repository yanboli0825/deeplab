"""Base datamodule utilities shared by framework datamodule implementations."""

from __future__ import annotations

import os
from abc import abstractmethod
from typing import Any, Dict, Optional

import lightning as L

from src.datamodules.split import SplitIndices, SplitProvider
from src.utils.misc import save_yaml


class BaseDataModule(L.LightningDataModule):
    """Framework datamodule base class.

    Split resolution is intentionally narrow: the datamodule either consumes a
    framework-level split provider or falls back to `_default_split()`.
    """

    def __init__(
        self,
        data_cfg: Dict[str, Any],
        split_provider: Optional[SplitProvider] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the shared datamodule state.

        Args:
            data_cfg: Datamodule-specific parameters used to load data.
            split_provider: Optional provider that can generate split indices on demand.
            *args: Extra positional arguments forwarded for compatibility.
            **kwargs: Extra keyword arguments forwarded for compatibility.

        Returns:
            None: The constructor initializes internal datamodule state.
        """

        super().__init__()
        self.save_hyperparameters(ignore=["split_provider"])
        self._split_provider = split_provider
        self._active_split: Optional[SplitIndices] = None


    def resolve_split(self) -> SplitIndices:
        """Resolve the split from provider or datamodule fallback.

        Returns:
            SplitIndices: Train/validation/test indices used by the datamodule.
        """

        provider = self._split_provider
        if provider is not None:
            return provider.build()

        return self._default_split()


    def emit_data_artifacts(self) -> None:
        """Persist split artifacts when runtime paths are available.

        Returns:
            None: The function writes the resolved split manifest when possible.
        """

        runtime = self.hparams.data_cfg.get("runtime", {})
        output_dir = runtime.get("output_dir")
        artifacts_cfg = runtime.get("artifacts", {})
        if not output_dir:
            return

        split_name = artifacts_cfg.get("split_manifest_name", "split_manifest.yaml")
        if self._active_split is not None:
            save_yaml(self._active_split.to_dict(), os.path.join(output_dir, split_name))


    def data_artifacts(self) -> Dict[str, Any]:
        """Return runtime data artifact paths.

        Returns:
            Dict[str, Any]: Split artifact paths exposed to the runtime artifact index.
        """

        runtime = self.hparams.data_cfg.get("runtime", {})
        output_dir = runtime.get("output_dir")
        artifacts_cfg = runtime.get("artifacts", {})
        split_path = None
        if output_dir:
            split_path = os.path.join(
                output_dir,
                artifacts_cfg.get("split_manifest_name", "split_manifest.yaml"),
            )

        return {
            "split_manifest": split_path,
        }


    @abstractmethod
    def _default_split(self) -> SplitIndices:
        """Build the fallback split used when no external split is provided.

        Returns:
            SplitIndices: Default split for the datamodule implementation.
        """

        raise NotImplementedError
