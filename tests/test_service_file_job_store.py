from service.app.storage.file_job_store import FileJobStore
from service.app.settings import ServiceSettings


def test_file_job_store_create_and_update(tmp_path) -> None:
    runtime_dir = tmp_path / "runtime"
    settings = ServiceSettings(
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
    store = FileJobStore(settings)

    created = store.create_job("train", {"framework": {}, "platform": {}})
    assert created["status"] == "QUEUED"

    updated = store.update_job(created["job_id"], {"status": "RUNNING"})
    assert updated["status"] == "RUNNING"

    fetched = store.get_job(created["job_id"])
    assert fetched["job_id"] == created["job_id"]
