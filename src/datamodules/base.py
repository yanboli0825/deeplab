"""Base datamodule utilities shared by framework datamodule implementations."""

from __future__ import annotations

import os
from abc import abstractmethod
from typing import Any, Dict, Optional

import lightning as L
import numpy as np
import yaml

from src.datamodules.split import SplitIndices, SplitProvider
from src.utils.misc import save_json, save_yaml


class BaseDataModule(L.LightningDataModule):
    """Framework datamodule base class.

    Datamodules consume split artifacts instead of owning split policy. They may
    optionally emit dataset metadata artifacts for downstream workflows.

    For split artifacts, we have two options: split indices and split provider. Split indices
    is prior than split provider. If they two are not provided, we automatically use default split.
    """

    def __init__(
        self,
        data_cfg: Dict[str, Any],
        split_indices: Optional[SplitIndices] = None,
        split_provider: Optional[SplitProvider] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the shared datamodule state.

        Args:
            data_cfg: Datamodule-specific parameters used to load data.
            split_indices: Optional precomputed split indices.
            split_provider: Optional provider that can generate split indices on demand.
            *args: Extra positional arguments forwarded for compatibility.
            **kwargs: Extra keyword arguments forwarded for compatibility.

        Returns:
            None: The constructor initializes internal datamodule state.
        """

        super().__init__()
        self.save_hyperparameters()
        self._active_split: Optional[SplitIndices] = None

    def resolve_split(self) -> SplitIndices:
        """Resolve the split from explicit indices, manifest, provider, or fallback.

        Returns:
            SplitIndices: Train/validation/test indices used by the datamodule.
        """

        split = self.hparams.split_indices
        if split is not None:
            return split

        split = self._load_split_manifest()
        if split is not None:
            return split

        provider = self.hparams.split_provider
        if provider is not None:
            return provider.build()

        return self._default_split()

    def _load_split_manifest(self) -> Optional[SplitIndices]:
        """Load split indices from a manifest file when one is configured.

        Returns:
            Optional[SplitIndices]: Parsed split indices, or `None` when no manifest is configured.
        """

        split_file = self.hparams.data_cfg.get("split_file")
        if not split_file:
            return None

        with open(split_file, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}

        return SplitIndices.from_mapping(payload)

    def emit_data_artifacts(self) -> None:
        """Persist split and dataset metadata artifacts when runtime paths are available.

        Returns:
            None: The function writes split and dataset metadata files when possible.
        """

        runtime = self.hparams.data_cfg.get("runtime", {})
        output_dir = runtime.get("output_dir")
        artifacts_cfg = runtime.get("artifacts", {})
        if not output_dir:
            return

        split_name = artifacts_cfg.get("split_manifest_name", "split_manifest.yaml")
        dataset_name = artifacts_cfg.get("dataset_summary_name", "dataset_summary.json")

        if self._active_split is not None:
            save_yaml(self._active_split.to_dict(), os.path.join(output_dir, split_name))
        save_json(self.dataset_summary(), os.path.join(output_dir, dataset_name))

    def dataset_summary(self) -> Dict[str, Any]:
        """Return datamodule-level metadata used by artifact indexing.

        Returns:
            Dict[str, Any]: Serializable metadata describing the dataset view used by the run.
        """

        total_samples = int(self.hparams.data_cfg.get("total_samples", 0))
        group_ids = self.hparams.data_cfg.get("group_ids")
        return {
            "dataset_name": self.hparams.data_cfg.get("dataset_name", self.__class__.__name__),
            "num_samples": total_samples,
            "input_shape": list(self.input_shape()),
            "label_space": self.label_space(),
            "groups_available": bool(group_ids),
        }

    def data_artifacts(self) -> Dict[str, Any]:
        """Return runtime data artifact paths and metadata.

        Returns:
            Dict[str, Any]: Artifact paths and metadata exposed to the runtime artifact index.
        """

        runtime = self.hparams.data_cfg.get("runtime", {})
        output_dir = runtime.get("output_dir")
        artifacts_cfg = runtime.get("artifacts", {})
        split_path = None
        dataset_summary_path = None
        if output_dir:
            split_path = os.path.join(
                output_dir,
                artifacts_cfg.get("split_manifest_name", "split_manifest.yaml"),
            )
            dataset_summary_path = os.path.join(
                output_dir,
                artifacts_cfg.get("dataset_summary_name", "dataset_summary.json"),
            )

        return {
            "split_manifest": split_path,
            "dataset_summary": dataset_summary_path,
            "metadata": self.dataset_summary(),
        }

    @abstractmethod
    def _default_split(self) -> SplitIndices:
        """Build the fallback split used when no external split is provided.

        Returns:
            SplitIndices: Default split for the datamodule implementation.
        """

        raise NotImplementedError

    @abstractmethod
    def input_shape(self) -> tuple[int, ...]:
        """Describe the shape of one input sample before batching.

        Returns:
            tuple[int, ...]: Sample input shape used for metadata only.
        """

        raise NotImplementedError

    @abstractmethod
    def label_space(self) -> Dict[str, Any]:
        """Describe the label space exposed by the datamodule.

        Returns:
            Dict[str, Any]: Serializable metadata about task type and label count.
        """

        raise NotImplementedError
