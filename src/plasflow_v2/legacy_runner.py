from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

from .constants import LEGACY_MODELS_DEFAULT, LEGACY_SCRIPT_DEFAULT


@dataclass
class LegacyRunResult:
    ok: bool
    used_mode: str
    stdout: str
    stderr: str
    reason: str | None = None


def run_legacy_classifier(
    input_path: Path,
    output_tsv: Path,
    threshold: float,
    script_path: Path | None = None,
    models_path: Path | None = None,
    timeout_sec: int = 60 * 60,
) -> LegacyRunResult:
    script = script_path or LEGACY_SCRIPT_DEFAULT
    models = models_path or LEGACY_MODELS_DEFAULT

    if not script.exists():
        return LegacyRunResult(
            ok=False,
            used_mode="v1",
            stdout="",
            stderr="",
            reason=f"Legacy script not found: {script}",
        )

    cmd = [
        sys.executable,
        str(script),
        "--input",
        str(input_path),
        "--output",
        str(output_tsv),
        "--threshold",
        str(threshold),
        "--models",
        str(models),
    ]

    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except Exception as exc:  # subprocess-level failure.
        return LegacyRunResult(
            ok=False,
            used_mode="v1",
            stdout="",
            stderr=str(exc),
            reason=f"Legacy runner invocation failed: {exc}",
        )

    if completed.returncode != 0:
        reason = (
            f"Legacy runner exited with code {completed.returncode}. "
            "Most common cause: unavailable v1 dependencies (TensorFlow 0.10 / R / rpy2 / Biostrings)."
        )
        return LegacyRunResult(
            ok=False,
            used_mode="v1",
            stdout=completed.stdout,
            stderr=completed.stderr,
            reason=reason,
        )

    return LegacyRunResult(
        ok=True,
        used_mode="v1",
        stdout=completed.stdout,
        stderr=completed.stderr,
        reason=None,
    )
