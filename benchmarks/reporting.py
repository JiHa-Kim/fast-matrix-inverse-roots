"""Centralized reporting and Markdown utilities for benchmarks."""

from __future__ import annotations

from typing import Any, Dict

from .utils import format_timestamp


def build_report_header(title: str, config: Dict[str, Any]) -> list[str]:
    """Generate a standardized Markdown report header with configuration."""
    lines = [
        f"# {title}",
        "",
        f"Generated: {format_timestamp()}",
        "",
        "## Run Configuration",
        "",
    ]
    for key in sorted(config.keys()):
        # Handle simple types for display
        val = config[key]
        if isinstance(val, bool):
            lines.append(f"- {key}: `{val}`")
        elif isinstance(val, (int, float, str)):
            lines.append(f"- {key}: `{val}`")
    lines.append("")
    return lines


def build_reproducibility_section(
    json_path: str, manifest_path: str, repo_root: str
) -> list[str]:
    """Generate a standardized reproducibility section."""
    import os

    def rel(p):
        return os.path.relpath(p, repo_root).replace("\\", "/")

    lines = [
        "## Reproducibility",
        "",
        "This report is paired with:",
        f"- `{rel(json_path)}` (raw per-step rows)",
        f"- `{rel(manifest_path)}` (run metadata + reproducibility fingerprint)",
        "- `.sha256` sidecars for all output files",
        "",
    ]
    return lines


def format_markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Standardize Markdown table formatting."""
    if not headers:
        return ""

    lines = []
    lines.append("| " + " | ".join(headers) + " |")

    # Simple alignment heuristic: right-align numbers, left-align text
    aligns = []
    for h in headers:
        aligns.append("---:")
    lines.append("| " + " | ".join(aligns) + " |")

    for row in rows:
        formatted_row = []
        for val in row:
            if isinstance(val, float):
                if abs(val) < 1e-3 and val != 0:
                    formatted_row.append(f"{val:.2e}")
                else:
                    formatted_row.append(f"{val:.4f}")
            else:
                formatted_row.append(str(val))
        lines.append("| " + " | ".join(formatted_row) + " |")

    return "\n".join(lines)
