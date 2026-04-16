from pathlib import Path

from service.app.engine.config_adapter import ConfigAdapter
from service.app.settings import ServiceSettings


def _settings(tmp_path: Path) -> ServiceSettings:
    runtime_dir = tmp_path / "runtime"
    return ServiceSettings(
        repo_root=tmp_path,
        runtime_dir=runtime_dir,
        jobs_dir=runtime_dir / "jobs",
        logs_dir=runtime_dir / "logs",
        generated_configs_dir=runtime_dir / "generated_configs",
        index_dir=runtime_dir / "index",
        python_executable="python",
        main_script=tmp_path / "main.py",
        max_concurrent_jobs=1,
        default_job_timeout_hours=1,
    )


def test_config_adapter_writes_manifest_and_injects_data_file(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    for path in (
        settings.jobs_dir,
        settings.logs_dir,
        settings.generated_configs_dir,
        settings.index_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    adapter = ConfigAdapter(settings)
    payload = {
        "platform": {"project_name": "demo"},
        "dataset": {
            "manifest_content": "case_id,label\nA,0\nB,1\n",
            "manifest_filename": "manifest.csv",
        },
        "framework": {
            "model": {"_target_": "src.models.cpath.abmil.ABMIL"},
            "datamodule": {
                "_target_": "src.datamodules.dummy_dm.DummyDataModule",
                "init_args": {"data_cfg": {"batch_size": 1}},
            },
            "trainer": {"_target_": "lightning.pytorch.Trainer"},
            "split": {"method": "holdout"},
        },
    }

    normalized_cfg, overrides, config_path = adapter.prepare(
        job_id="job_test",
        job_type="train",
        request_payload=payload,
    )

    manifest_path = Path(normalized_cfg["split"]["data_file"])
    assert manifest_path.exists()
    assert normalized_cfg["datamodule"]["init_args"]["data_cfg"]["data_file"] == str(manifest_path)
    assert any(item.startswith("split.data_file=") for item in overrides)
    assert Path(config_path).exists()
