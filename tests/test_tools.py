from __future__ import annotations

from plasflow_v2.tools import probe_tools


def test_probe_tools_schema() -> None:
    result = probe_tools()
    for name in ("racon", "medaka", "prodigal", "blastn", "diamond"):
        assert name in result
        assert "available" in result[name]
        assert "path" in result[name]
