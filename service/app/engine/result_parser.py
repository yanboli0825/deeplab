"""Parse framework artifacts into service-facing summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from service.app.domain.exceptions import InvalidJobRequestError


class ResultParser:
    """Read framework-owned summary files after a job completes."""

    def parse(self, *, job_type: str, job_root: Path) -> Dict[str, Any]:
        if job_type == "train":
            return self._parse_single_run(job_root / "run")
        if job_type == "cv":
            return self._parse_cv_run(job_root / "run")
        if job_type == "hpo":
            return self._parse_hpo_run(job_root / "sweep")
        raise InvalidJobRequestError(f"Unsupported job type '{job_type}'")

    def _parse_single_run(self, output_dir: Path) -> Dict[str, Any]:
        summary_path = output_dir / "run_summary.json"
        artifact_path = output_dir / "artifacts.json"
        summary = self._read_json(summary_path)
        return {
            "output_dir": str(output_dir),
            "summary_path": str(summary_path),
            "artifact_index_path": str(artifact_path),
            "monitor": summary.get("monitor"),
            "val_score": summary.get("val_score"),
            "test_score": summary.get("test_score"),
            "best_ckpt_path": summary.get("best_ckpt_path"),
            "best_metrics": summary.get("best_metrics", {}),
            "summary": summary,
        }

    def _parse_cv_run(self, output_dir: Path) -> Dict[str, Any]:
        summary_path = output_dir / "workflow_summary.json"
        artifact_path = output_dir / "workflow_artifacts.json"
        summary = self._read_json(summary_path)
        return {
            "output_dir": str(output_dir),
            "summary_path": str(summary_path),
            "artifact_index_path": str(artifact_path),
            "monitor": summary.get("monitor"),
            "val_score": summary.get("val_score"),
            "test_score": summary.get("test_score"),
            "best_ckpt_path": None,
            "best_metrics": {},
            "summary": summary,
        }

    def _parse_hpo_run(self, sweep_dir: Path) -> Dict[str, Any]:
        trial_records: List[Dict[str, Any]] = []
        direction = "minimize"
        for child in sorted(sweep_dir.iterdir()):
            if not child.is_dir():
                continue
            summary_path = child / "workflow_summary.json"
            if not summary_path.exists():
                continue
            summary = self._read_json(summary_path)
            trial_records.append(
                {
                    "trial_dir": str(child),
                    "summary_path": str(summary_path),
                    "val_score": summary.get("val_score"),
                    "test_score": summary.get("test_score"),
                    "monitor": summary.get("monitor"),
                }
            )
        if not trial_records:
            raise InvalidJobRequestError(f"No HPO trial summaries found under {sweep_dir}")
        best_trial = self._select_best_trial(trial_records, direction=direction)
        aggregate = {
            "direction": direction,
            "best_trial": best_trial,
            "trials": trial_records,
        }
        summary_path = sweep_dir / "hpo_summary.json"
        summary_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
        return {
            "output_dir": str(sweep_dir),
            "summary_path": str(summary_path),
            "artifact_index_path": None,
            "monitor": best_trial.get("monitor"),
            "val_score": best_trial.get("val_score"),
            "test_score": best_trial.get("test_score"),
            "best_ckpt_path": None,
            "best_metrics": {},
            "summary": aggregate,
        }

    @staticmethod
    def _select_best_trial(trials: List[Dict[str, Any]], *, direction: str) -> Dict[str, Any]:
        key_fn = lambda item: float(item["val_score"])
        return min(trials, key=key_fn) if direction == "minimize" else max(trials, key=key_fn)

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise InvalidJobRequestError(f"Expected summary file does not exist: {path}")
        return json.loads(path.read_text(encoding="utf-8"))
