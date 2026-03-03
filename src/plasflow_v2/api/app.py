from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from ..constants import DEFAULT_RUNS_DIR, DEFAULT_THRESHOLD, normalize_mode
from ..pipeline import artifact_paths
from .db import create_job, get_job, init_db, update_job
from .worker import run_job


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _artifact_map(prefix: Path) -> dict[str, Path]:
    artifacts = artifact_paths(prefix)
    return {
        "tsv": artifacts.tsv,
        "plasmids.fasta": artifacts.plasmids_fasta,
        "chromosomes.fasta": artifacts.chromosomes_fasta,
        "unclassified.fasta": artifacts.unclassified_fasta,
        "phage.fasta": artifacts.phage_fasta,
        "ambiguous.fasta": artifacts.ambiguous_fasta,
        "report.json": artifacts.report_json,
        "report.html": artifacts.report_html,
    }


def _done_callback(db_path: Path, futures: dict[str, Future], job_id: str, fut: Future) -> None:
    finished_at = _now_iso()
    try:
        payload = fut.result()
        update_job(
            db_path,
            job_id,
            status="completed",
            progress=1.0,
            used_mode=payload.get("used_mode"),
            fallback_reason=payload.get("fallback_reason"),
            finished_at=finished_at,
        )
    except Exception as exc:
        update_job(
            db_path,
            job_id,
            status="failed",
            progress=1.0,
            error=str(exc),
            finished_at=finished_at,
        )
    finally:
        futures.pop(job_id, None)


def create_app() -> FastAPI:
    runs_dir = Path(os.getenv("PLASFLOW_RUNS_DIR", str(DEFAULT_RUNS_DIR))).resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    db_path = runs_dir / "plasflow_jobs.db"

    app = FastAPI(title="PlasFlow v2 API", version="2.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    init_db(db_path)

    executor_mode = os.getenv("PLASFLOW_EXECUTOR", "process")
    max_workers = int(os.getenv("PLASFLOW_MAX_WORKERS", "2"))
    executor: ProcessPoolExecutor | None = None
    if executor_mode != "inline":
        executor = ProcessPoolExecutor(max_workers=max_workers)

    app.state.db_path = db_path
    app.state.runs_dir = runs_dir
    app.state.executor_mode = executor_mode
    app.state.executor = executor
    app.state.futures: dict[str, Future] = {}

    @app.get("/api/v1/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "executor_mode": app.state.executor_mode,
            "runs_dir": str(app.state.runs_dir),
        }

    @app.post("/api/v1/jobs")
    async def create_new_job(
        file: UploadFile = File(...),
        mode: str = Form("v1"),
        task: str = Form("legacy28"),
        threshold: float = Form(DEFAULT_THRESHOLD),
        read_type: str = Form("short"),
        min_length: int = Form(1000),
        coverage_source: str = Form("header"),
        circularity_check: bool = Form(True),
        polish: str = Form("none"),
    ) -> dict[str, Any]:
        try:
            normalized_mode = normalize_mode(mode)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if task not in {"legacy28", "binary_domain", "domain4"}:
            raise HTTPException(status_code=400, detail="task must be legacy28|binary_domain|domain4")
        if read_type not in {"short", "long", "hybrid", "complete"}:
            raise HTTPException(status_code=400, detail="read_type must be short|long|hybrid|complete")
        if coverage_source not in {"header", "none"}:
            raise HTTPException(status_code=400, detail="coverage_source must be header|none")
        if polish not in {"none", "racon", "medaka"}:
            raise HTTPException(status_code=400, detail="polish must be none|racon|medaka")

        job_id = str(uuid4())
        job_dir = app.state.runs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        filename = file.filename or "input.fasta"
        input_path = job_dir / filename
        data = await file.read()
        input_path.write_bytes(data)

        output_prefix = job_dir / "result"
        created_at = _now_iso()

        create_job(
            app.state.db_path,
            {
                "job_id": job_id,
                "status": "queued",
                "progress": 0.0,
                "mode": normalized_mode,
                "task": task,
                "threshold": float(threshold),
                "read_type": read_type,
                "min_length": int(min_length),
                "coverage_source": coverage_source,
                "circularity_check": int(bool(circularity_check)),
                "polish": polish,
                "input_path": str(input_path),
                "output_prefix": str(output_prefix),
                "error": None,
                "requested_mode": normalized_mode,
                "used_mode": None,
                "fallback_reason": None,
                "created_at": created_at,
                "started_at": None,
                "finished_at": None,
            },
        )

        if app.state.executor_mode == "inline":
            update_job(app.state.db_path, job_id, status="running", progress=0.2, started_at=_now_iso())
            try:
                payload = run_job(
                    job_id=job_id,
                    input_path=str(input_path),
                    output_prefix=str(output_prefix),
                    mode=normalized_mode,
                    task=task,
                    threshold=float(threshold),
                    read_type=read_type,
                    min_length=int(min_length),
                    coverage_source=coverage_source,
                    circularity_check=bool(circularity_check),
                    polish=polish,
                )
                update_job(
                    app.state.db_path,
                    job_id,
                    status="completed",
                    progress=1.0,
                    used_mode=payload.get("used_mode"),
                    fallback_reason=payload.get("fallback_reason"),
                    finished_at=_now_iso(),
                )
            except Exception as exc:
                update_job(
                    app.state.db_path,
                    job_id,
                    status="failed",
                    progress=1.0,
                    error=str(exc),
                    finished_at=_now_iso(),
                )
        else:
            update_job(app.state.db_path, job_id, status="running", progress=0.2, started_at=_now_iso())
            future = app.state.executor.submit(
                run_job,
                job_id,
                str(input_path),
                str(output_prefix),
                normalized_mode,
                task,
                float(threshold),
                read_type,
                int(min_length),
                coverage_source,
                bool(circularity_check),
                polish,
            )
            app.state.futures[job_id] = future
            future.add_done_callback(
                lambda fut, jid=job_id: _done_callback(app.state.db_path, app.state.futures, jid, fut)
            )

        return {"job_id": job_id}

    @app.get("/api/v1/jobs/{job_id}")
    def get_job_status(job_id: str) -> dict[str, Any]:
        job = get_job(app.state.db_path, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        return job

    @app.get("/api/v1/jobs/{job_id}/artifacts")
    def list_job_artifacts(job_id: str) -> dict[str, Any]:
        job = get_job(app.state.db_path, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")

        mapping = _artifact_map(Path(job["output_prefix"]))
        artifacts = []
        for name, path in mapping.items():
            if path.exists():
                artifacts.append({"name": name, "path": str(path), "size_bytes": path.stat().st_size})

        return {"job_id": job_id, "artifacts": artifacts}

    @app.get("/api/v1/jobs/{job_id}/download/{artifact_name}")
    def download_artifact(job_id: str, artifact_name: str):
        job = get_job(app.state.db_path, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")

        mapping = _artifact_map(Path(job["output_prefix"]))
        if artifact_name not in mapping:
            raise HTTPException(status_code=404, detail="unknown artifact")

        file_path = mapping[artifact_name]
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="artifact not ready")

        return FileResponse(path=str(file_path), filename=file_path.name)

    return app
