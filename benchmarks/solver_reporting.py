"""Standardized Markdown reporting for solver benchmarks."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List

from .utils import clean_method_name
from .solver_utils import ParsedRow, assessment_score
from .reporting import build_report_header


def _build_legend(ab_mode: bool = False) -> list[str]:
    """Build a standardized legend for benchmark reports."""
    res = [
        "---",
        "",
        "### Legend",
        "- **Bold values** indicate the best performer for that metric in the scenario." if not ab_mode else "- Metrics are compared between Side A and Side B.",
        "- `total_ms`: Total execution time including preprocessing.",
        "- `iter_ms`: Time spent in iterations.",
        "- `relerr`: Median relative error vs ground truth (for SPD) or reference solver (for Non-SPD).",
        "- `relerr_p90`: 90th percentile relative error (tail quality).",
        "- `resid`: Median residual error (||Ax - b|| / ||b||).",
        "- `resid_p90`: 90th percentile residual error.",
        "- `fail_rate`: Fraction of trials that were non-finite or failed quality checks.",
        "- `q_per_ms`: Quality (digits of precision, i.e., -log10(relerr)) per millisecond of compute.",
    ]
    if ab_mode:
        res.extend([
            "- `delta_ms`: Change in total milliseconds (B - A).",
            "- `delta_pct`: Percentage change in total milliseconds relative to A.",
            "- `ratio(B/A)`: Ratio of B's metric to A's metric. < 1.0 means B is smaller/better for errors/time.",
        ])
    res.append("")
    return res


def to_markdown(
    all_rows: List[ParsedRow],
    *,
    config: Dict[str, Any] | None = None,
) -> str:
    """Generate a full solver benchmark markdown report."""

    def internal_clean(n: str) -> str:
        return clean_method_name(n).replace("-Reuse", "-R")

    out: list[str] = build_report_header("Solver Benchmark Report", config or {})
    out.extend(_build_legend(ab_mode=False))

    # Hierarchy: Kind -> p -> (n, k) -> case -> Methods
    kind_groups = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for row in all_rows:
        kind, p, n, k, case = row[0], row[1], row[2], row[3], row[4]
        kind_groups[kind][p][(n, k)][case].append(row)

    for kind in sorted(kind_groups.keys()):
        kind_label = "SPD" if kind == "spd" else "Non-Normal"
        out.append(f"# {kind_label}")
        out.append("")

        for p in sorted(kind_groups[kind].keys()):
            out.append(f"## p = {p}")
            out.append("")

            for (n, k) in sorted(kind_groups[kind][p].keys()):
                out.append(f"### Size {n}x{n} | RHS {n}x{k}")
                out.append("")

                for case in sorted(kind_groups[kind][p][(n, k)].keys()):
                    out.append(f"#### Case: `{case}`")
                    out.append("")
                    out.append(
                        "| method | total_ms | iter_ms | relerr | relerr_p90 | resid | resid_p90 | fail_rate | q_per_ms |"
                    )
                    out.append("|:---|---:|---:|---:|---:|---:|---:|---:|---:|")

                    rows = kind_groups[kind][p][(n, k)][case]

                    # Find bests for this block
                    best_total = min(r[6] for r in rows)
                    best_iter = min(r[7] for r in rows)
                    best_relerr = min(r[8] for r in rows)
                    best_relerr_p90 = min(r[9] for r in rows)
                    best_fail = min(r[12] for r in rows)
                    best_qpm = max(r[13] for r in rows)
                    # residuals might be nan if not spd or not requested
                    resids = [r[14] for r in rows if not math.isnan(r[14])]
                    best_resid = min(resids) if resids else float("nan")
                    resids_p90 = [r[15] for r in rows if not math.isnan(r[15])]
                    best_resid_p90 = min(resids_p90) if resids_p90 else float("nan")

                    def fmt(val, best, s, is_max=False, is_fail=False):
                        if is_fail and val >= 1.0:
                            return s
                        is_best = (val <= best) if not is_max else (val >= best)
                        if is_best and not math.isnan(val):
                            return f"**{s}**"
                        return s

                    for row in sorted(rows, key=lambda r: r[6]):
                        method = internal_clean(row[5])
                        total_ms = row[6]
                        iter_ms = row[7]
                        relerr = row[8]
                        relerr_p90 = row[9]
                        fail_rate = row[12]
                        qpm = row[13]
                        resid = row[14]
                        resid_p90 = row[15]

                        s_total = fmt(total_ms, best_total, f"{total_ms:.3f}")
                        s_iter = fmt(iter_ms, best_iter, f"{iter_ms:.3f}")
                        s_rel = fmt(relerr, best_relerr, f"{relerr:.2e}")
                        s_rel_p90 = fmt(relerr_p90, best_relerr_p90, f"{relerr_p90:.2e}")
                        s_fail = fmt(fail_rate, best_fail, f"{100.0*fail_rate:.1f}%", is_fail=True)
                        s_qpm = fmt(qpm, best_qpm, f"{qpm:.3e}", is_max=True)
                        s_resid = fmt(resid, best_resid, f"{resid:.2e}")
                        s_resid_p90 = fmt(resid_p90, best_resid_p90, f"{resid_p90:.2e}")

                        out.append(
                            f"| {method} | {s_total} | {s_iter} | {s_rel} | {s_rel_p90} | "
                            f"{s_resid} | {s_resid_p90} | {s_fail} | {s_qpm} |"
                        )
                    out.append("")
    
    return "\n".join(out)


def to_markdown_ab(
    rows_a: List[ParsedRow],
    rows_b: List[ParsedRow],
    *,
    label_a: str,
    label_b: str,
    match_on_method: bool,
    config: Dict[str, Any] | None = None,
) -> str:
    """Generate an A/B solver benchmark markdown comparison."""

    def _key_method(row: ParsedRow):
        return row[0], row[1], row[2], row[3], row[4], row[5]

    def _key_case(row: ParsedRow):
        return row[0], row[1], row[2], row[3], row[4]

    def _build_index(
        rows: list[ParsedRow], use_method_key: bool
    ) -> dict[Any, ParsedRow]:
        out_idx: dict[Any, ParsedRow] = {}
        for r in rows:
            key = _key_method(r) if use_method_key else _key_case(r)
            if key in out_idx:
                raise RuntimeError(
                    f"A/B compare has duplicate rows per match key: {key}. "
                    "Use --methods to keep one method per side, or enable "
                    "--ab-match-on-method when comparing like-for-like methods."
                )
            out_idx[key] = r
        return out_idx

    map_a = _build_index(rows_a, use_method_key=match_on_method)
    map_b = _build_index(rows_b, use_method_key=match_on_method)
    keys = sorted(set(map_a.keys()) & set(map_b.keys()))

    if len(keys) == 0:
        raise RuntimeError(
            "A/B rows had no overlapping keys; cannot build comparable report."
        )

    out: list[str] = build_report_header("Solver Benchmark A/B Report", config or {})
    out.extend(_build_legend(ab_mode=True))

    out.extend(
        [
            f"A: {label_a}",
            f"B: {label_b}",
            "",
        ]
    )

    if match_on_method:
        out.append(
            "| kind | p | n | k | case | method | "
            f"{label_a}_total_ms | {label_b}_total_ms | delta_ms(B-A) | delta_pct | "
            f"{label_a}_iter_ms | {label_b}_iter_ms | "
            f"{label_a}_relerr | {label_b}_relerr | relerr_ratio(B/A) | "
            f"{label_a}_relerr_p90 | {label_b}_relerr_p90 | "
            f"{label_a}_fail_rate | {label_b}_fail_rate | "
            f"{label_a}_q_per_ms | {label_b}_q_per_ms | q_per_ms_ratio(B/A) |"
        )
        out.append(
            "|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
    else:
        out.append(
            "| kind | p | n | k | case | "
            f"{label_a}_method | {label_b}_method | "
            f"{label_a}_total_ms | {label_b}_total_ms | delta_ms(B-A) | delta_pct | "
            f"{label_a}_iter_ms | {label_b}_iter_ms | "
            f"{label_a}_relerr | {label_b}_relerr | relerr_ratio(B/A) | "
            f"{label_a}_relerr_p90 | {label_b}_relerr_p90 | "
            f"{label_a}_fail_rate | {label_b}_fail_rate | "
            f"{label_a}_q_per_ms | {label_b}_q_per_ms | q_per_ms_ratio(B/A) |"
        )
        out.append(
            "|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        )

    b_faster = 0
    b_better_quality = 0
    b_better_score = 0

    for key in keys:
        ra = map_a[key]
        rb = map_b[key]
        kind, p_val, n, k, case_name = ra[0], ra[1], ra[2], ra[3], ra[4]
        method_a = ra[5]
        method_b = rb[5]
        a_total, a_iter, a_rel = ra[6], ra[7], ra[8]
        b_total, b_iter, b_rel = rb[6], rb[7], rb[8]
        a_rel_p90, a_fail, a_qpm = ra[9], ra[10], ra[11]
        b_rel_p90, b_fail, b_qpm = rb[9], rb[10], rb[11]

        d_ms = b_total - a_total
        d_pct = (100.0 * d_ms / a_total) if a_total != 0 else float("nan")
        rel_ratio = (b_rel / a_rel) if a_rel != 0 else float("nan")
        qpm_ratio = (b_qpm / a_qpm) if a_qpm != 0 else float("nan")

        if rb[6] < ra[6]:
            b_faster += 1
        if rb[8] <= ra[8] and rb[9] <= ra[9] and rb[10] <= ra[10]:
            b_better_quality += 1
        if assessment_score(rb) > assessment_score(ra):
            b_better_score += 1

        if match_on_method:
            out.append(
                f"| {kind} | {p_val} | {n} | {k} | {case_name} | {method_a} | "
                f"{a_total:.3f} | {b_total:.3f} | {d_ms:.3f} | {d_pct:.2f}% | "
                f"{a_iter:.3f} | {b_iter:.3f} | {a_rel:.3e} | {b_rel:.3e} | {rel_ratio:.3f} | "
                f"{a_rel_p90:.3e} | {b_rel_p90:.3e} | {100.0 * a_fail:.1f}% | {100.0 * b_fail:.1f}% | "
                f"{a_qpm:.3e} | {b_qpm:.3e} | {qpm_ratio:.3f} |"
            )
        else:
            out.append(
                f"| {kind} | {p_val} | {n} | {k} | {case_name} | {method_a} | {method_b} | "
                f"{a_total:.3f} | {b_total:.3f} | {d_ms:.3f} | {d_pct:.2f}% | "
                f"{a_iter:.3f} | {b_iter:.3f} | {a_rel:.3e} | {b_rel:.3e} | {rel_ratio:.3f} | "
                f"{a_rel_p90:.3e} | {b_rel_p90:.3e} | {100.0 * a_fail:.1f}% | {100.0 * b_fail:.1f}% | "
                f"{a_qpm:.3e} | {b_qpm:.3e} | {qpm_ratio:.3f} |"
            )

    total = len(keys)
    out.append("")
    out.append("## A/B Summary")
    out.append("")
    out.append("| metric | count | share |")
    out.append("|---|---:|---:|")
    out.append(
        f"| B faster (total_ms) | {b_faster} / {total} | "
        f"{(100.0 * b_faster / total) if total > 0 else 0.0:.1f}% |"
    )
    out.append(
        f"| B better-or-equal quality (`relerr`,`relerr_p90`,`fail_rate`) | "
        f"{b_better_quality} / {total} | "
        f"{(100.0 * b_better_quality / total) if total > 0 else 0.0:.1f}% |"
    )
    out.append(
        f"| B better assessment score | {b_better_score} / {total} | "
        f"{(100.0 * b_better_score / total) if total > 0 else 0.0:.1f}% |"
    )
    out.append("")
    return "\n".join(out)
