from __future__ import annotations

from shutil import which
from typing import Any


OPTIONAL_EXTERNAL_TOOLS = ("racon", "medaka", "prodigal", "blastn", "diamond")


def probe_tools() -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for name in OPTIONAL_EXTERNAL_TOOLS:
        path = which(name)
        results[name] = {
            "available": path is not None,
            "path": path,
        }
    return results
