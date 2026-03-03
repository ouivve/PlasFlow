from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from plasflow_v2.api.app import create_app


def test_api_job_lifecycle_inline(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PLASFLOW_EXECUTOR", "inline")
    monkeypatch.setenv("PLASFLOW_RUNS_DIR", str(tmp_path / "runs"))

    app = create_app()
    client = TestClient(app)

    fasta = tmp_path / "input.fasta"
    fasta.write_text(">n1\n" + "ACGT" * 80 + "\n>n2\n" + "ATAT" * 90 + "\n", encoding="utf-8")

    with fasta.open("rb") as handle:
        response = client.post(
            "/api/v1/jobs",
            files={"file": ("input.fasta", handle, "application/octet-stream")},
            data={"mode": "v2", "task": "legacy28", "threshold": "0.7"},
        )

    assert response.status_code == 200
    job_id = response.json()["job_id"]

    status_resp = client.get(f"/api/v1/jobs/{job_id}")
    assert status_resp.status_code == 200
    status = status_resp.json()
    assert status["status"] in {"completed", "failed"}

    artifacts_resp = client.get(f"/api/v1/jobs/{job_id}/artifacts")
    assert artifacts_resp.status_code == 200
    artifacts = artifacts_resp.json()["artifacts"]
    if status["status"] == "completed":
        assert any(item["name"] == "tsv" for item in artifacts)
