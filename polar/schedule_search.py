from __future__ import annotations

import dataclasses
from typing import List, Sequence, Tuple

from polar.zolo import (
    ZoloCoeffs,
    dwh_coeffs_from_ell,
    dwh_ell_next,
    temper_zolo_coeffs_by_floor,
    zolo_coeffs_from_ell,
    zolo_ell_next,
    zolo_safe_for_cholesky,
)


@dataclasses.dataclass(frozen=True)
class StepSpec:
    kind: str
    ell_in: float
    ell_out: float
    pred_kappa_after: float
    r: int = 0
    pole_floor: float = 0.0


@dataclasses.dataclass
class ScheduleInfo:
    objective_name: str
    cost_key: Tuple[float, ...]
    steps: List[StepSpec]


def dwh_stability_metrics(ell: float) -> Tuple[float, float]:
    a, b, c = dwh_coeffs_from_ell(ell)
    ell2 = float(ell) * float(ell)
    return float((1.0 + c) / (ell2 + c)), float(max(abs(b / c), abs(a - b / c), 1.0))


def zolo_stability_metrics(ell: float, coeffs: ZoloCoeffs) -> Tuple[float, float]:
    ell2 = float(ell) * float(ell)
    solve_cond = float(max((1.0 + c) / (ell2 + c) for c in coeffs.c_odd))
    combine_scale = float(max(max(abs(v) for v in coeffs.a), abs(coeffs.mhat), 1.0))
    return solve_cond, combine_scale


def zolo_badness_metrics(ell: float, coeffs: ZoloCoeffs) -> Tuple[float, float, float]:
    ell2 = float(ell) * float(ell)
    max_cond = float(max((1.0 + c) / (ell2 + c) for c in coeffs.c_odd))
    max_a = float(max(abs(v) for v in coeffs.a))
    cancel = float(sum(abs(v) for v in coeffs.a))
    return max_cond, max_a, cancel


def step_small_solve_count(step: StepSpec) -> int:
    return 1 if step.kind == "DWH" else int(step.r)


def tempered_zolo_candidate(
    ell: float,
    r: int,
    pole_floor: float,
    dps: int,
) -> Tuple[StepSpec, ZoloCoeffs, Tuple[float, float, float]]:
    coeffs = zolo_coeffs_from_ell(int(r), float(ell), dps=int(dps))
    coeffs = temper_zolo_coeffs_by_floor(coeffs, pole_floor=float(pole_floor))
    ell_out = zolo_ell_next(float(ell), coeffs)
    max_cond, max_a, cancel = zolo_badness_metrics(float(ell), coeffs)
    step = StepSpec(
        kind="TZOLO",
        ell_in=float(ell),
        ell_out=float(ell_out),
        pred_kappa_after=float(1.0 / max(ell_out, 1e-300)),
        r=int(r),
        pole_floor=float(max(pole_floor, 0.0)),
    )
    return step, coeffs, (max_cond, max_a, cancel)


def solve_optimal_schedule_exact(
    ell0: float,
    target_kappa_O: float,
    max_steps: int,
    zolo_r_values: Sequence[int],
    zolo_coeff_dps: int,
    zolo_shift_cond_max: float,
    zolo_max_a: float,
) -> ScheduleInfo:
    memo = {}

    def key_ell(x: float) -> float:
        return float(f"{x:.16e}")

    def rec(ell: float, steps_left: int):
        ell = key_ell(ell)
        if 1.0 / max(ell, 1e-300) <= float(target_kappa_O):
            return (0, 0, 1.0 / max(ell, 1e-300)), []
        if steps_left == 0:
            return (10**9, 10**9, float("inf")), []

        key = (ell, steps_left)
        if key in memo:
            return memo[key]

        best_cost = (10**9, 10**9, float("inf"))
        best_sched: List[StepSpec] = []

        ell_dwh = dwh_ell_next(ell)
        step = StepSpec("DWH", float(ell), float(ell_dwh), float(1.0 / max(ell_dwh, 1e-300)), 1)
        rem_cost, rem_sched = rec(ell_dwh, steps_left - 1)
        cand_cost = (1 + rem_cost[0], 1 + rem_cost[1], rem_cost[2])
        if cand_cost < best_cost:
            best_cost = cand_cost
            best_sched = [step] + rem_sched

        for r in sorted(int(v) for v in zolo_r_values):
            coeffs = zolo_coeffs_from_ell(int(r), float(ell), dps=int(zolo_coeff_dps))
            if not zolo_safe_for_cholesky(
                float(ell), coeffs, zolo_shift_cond_max, zolo_max_a
            ):
                continue
            ell_z = zolo_ell_next(float(ell), coeffs)
            step = StepSpec("ZOLO", float(ell), float(ell_z), float(1.0 / max(ell_z, 1e-300)), int(r))
            rem_cost, rem_sched = rec(ell_z, steps_left - 1)
            cand_cost = (1 + rem_cost[0], int(r) + rem_cost[1], rem_cost[2])
            if cand_cost < best_cost:
                best_cost = cand_cost
                best_sched = [step] + rem_sched

        memo[key] = (best_cost, best_sched)
        return memo[key]

    cost, sched = rec(float(ell0), int(max_steps))
    return ScheduleInfo("exact_optimal", cost, sched)


def solve_two_step_exact_cholesky(
    ell0: float,
    target_kappa_O: float,
    zolo_r_values: Sequence[int],
    zolo_coeff_dps: int,
    tighten_fraction: float,
) -> ScheduleInfo:
    tighten_fraction = float(min(max(tighten_fraction, 0.0), 1.0))
    effective_target = float(
        1.0 + (target_kappa_O - 1.0) * (1.0 - tighten_fraction)
    )

    if 1.0 / max(ell0, 1e-300) <= effective_target:
        return ScheduleInfo(
            "two_step_exact_cholesky",
            (0.0, 1.0 / max(ell0, 1e-300), 0.0),
            [],
        )

    best_key = None
    best_sched = None
    orders = sorted(int(v) for v in zolo_r_values)
    for r1 in orders:
        coeffs1 = zolo_coeffs_from_ell(r1, ell0, dps=int(zolo_coeff_dps))
        ell1 = zolo_ell_next(ell0, coeffs1)
        step1 = StepSpec("ZOLO", ell0, float(ell1), float(1.0 / max(ell1, 1e-300)), r1)
        if step1.pred_kappa_after <= effective_target:
            key = (float(r1), float(step1.pred_kappa_after), 0.0)
            if best_key is None or key < best_key:
                best_key = key
                best_sched = [step1]
            continue

        for r2 in orders:
            coeffs2 = zolo_coeffs_from_ell(r2, ell1, dps=int(zolo_coeff_dps))
            ell2 = zolo_ell_next(ell1, coeffs2)
            step2 = StepSpec("ZOLO", float(ell1), float(ell2), float(1.0 / max(ell2, 1e-300)), r2)
            if step2.pred_kappa_after > effective_target:
                continue
            key = (float(r1 + r2), float(step2.pred_kappa_after), float(r2))
            if best_key is None or key < best_key:
                best_key = key
                best_sched = [step1, step2]

    if best_key is None or best_sched is None:
        return ScheduleInfo(
            "two_step_exact_cholesky",
            (float("inf"), float("inf"), float("inf")),
            [],
        )
    return ScheduleInfo("two_step_exact_cholesky", best_key, best_sched)


def enumerate_safe_scalar_steps(
    ell: float,
    zolo_r_values: Sequence[int],
    zolo_coeff_dps: int,
    zolo_shift_cond_max: float,
    zolo_max_a: float,
) -> List[Tuple[StepSpec, float, float]]:
    out: List[Tuple[StepSpec, float, float]] = []
    ell_dwh = dwh_ell_next(ell)
    dwh_cond, dwh_scale = dwh_stability_metrics(ell)
    out.append(
        (
            StepSpec("DWH", float(ell), float(ell_dwh), float(1.0 / max(ell_dwh, 1e-300)), 1),
            dwh_cond,
            dwh_scale,
        )
    )

    for r in sorted(int(v) for v in zolo_r_values):
        coeffs = zolo_coeffs_from_ell(int(r), float(ell), dps=int(zolo_coeff_dps))
        if not zolo_safe_for_cholesky(
            float(ell), coeffs, zolo_shift_cond_max, zolo_max_a
        ):
            continue
        ell_z = zolo_ell_next(float(ell), coeffs)
        z_cond, z_scale = zolo_stability_metrics(float(ell), coeffs)
        out.append(
            (
                StepSpec("ZOLO", float(ell), float(ell_z), float(1.0 / max(ell_z, 1e-300)), int(r)),
                z_cond,
                z_scale,
            )
        )
    return out


def solve_two_step_schedule_sufficient(
    ell0: float,
    target_kappa_O: float,
    zolo_r_values: Sequence[int],
    zolo_coeff_dps: int,
    zolo_shift_cond_max: float,
    zolo_max_a: float,
    switch_ell_min: float,
) -> ScheduleInfo:
    if 1.0 / max(float(ell0), 1e-300) <= float(target_kappa_O):
        return ScheduleInfo(
            "two_step_sufficient",
            (1.0, 1.0, 0.0, 1.0 / max(float(ell0), 1e-300)),
            [],
        )

    first_steps = enumerate_safe_scalar_steps(
        ell=float(ell0),
        zolo_r_values=zolo_r_values,
        zolo_coeff_dps=zolo_coeff_dps,
        zolo_shift_cond_max=zolo_shift_cond_max,
        zolo_max_a=zolo_max_a,
    )

    def pick(require_switch: bool):
        best_score = None
        best_sched = None
        for step1, cond1, scale1 in first_steps:
            if require_switch and step1.ell_out < float(switch_ell_min):
                continue
            if step1.pred_kappa_after <= float(target_kappa_O):
                score = (
                    float(cond1),
                    float(scale1),
                    float(step_small_solve_count(step1)),
                    float(step1.pred_kappa_after),
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_sched = [step1]
                continue
            second_steps = enumerate_safe_scalar_steps(
                ell=float(step1.ell_out),
                zolo_r_values=zolo_r_values,
                zolo_coeff_dps=zolo_coeff_dps,
                zolo_shift_cond_max=zolo_shift_cond_max,
                zolo_max_a=zolo_max_a,
            )
            for step2, cond2, scale2 in second_steps:
                if step2.pred_kappa_after > float(target_kappa_O):
                    continue
                score = (
                    float(max(cond1, cond2)),
                    float(max(scale1, scale2)),
                    float(step_small_solve_count(step1) + step_small_solve_count(step2)),
                    float(step2.pred_kappa_after),
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_sched = [step1, step2]
        if best_score is None or best_sched is None:
            return None
        return best_score, best_sched

    picked = pick(require_switch=True)
    if picked is None:
        picked = pick(require_switch=False)
    if picked is None:
        return ScheduleInfo(
            "two_step_sufficient",
            (float("inf"), float("inf"), float("inf"), float("inf")),
            [],
        )
    return ScheduleInfo("two_step_sufficient", picked[0], picked[1])


def solve_two_step_tempered_cholesky(
    ell0: float,
    target_kappa_O: float,
    zolo_r_values: Sequence[int],
    zolo_coeff_dps: int,
    pole_floor_values: Sequence[float],
    max_cond_allowed: float,
    max_a_allowed: float,
    max_cancel_allowed: float,
    tighten_fraction: float,
    require_monotone_gain: bool = True,
) -> ScheduleInfo:
    tighten_fraction = float(min(max(tighten_fraction, 0.0), 1.0))
    effective_target = float(
        1.0 + (target_kappa_O - 1.0) * (1.0 - tighten_fraction)
    )

    if 1.0 / max(float(ell0), 1e-300) <= effective_target:
        return ScheduleInfo(
            "two_step_tempered_cholesky",
            (0.0, 0.0, 0.0, 0.0, 1.0 / max(float(ell0), 1e-300)),
            [],
        )

    best_key = None
    best_sched = None
    floors = sorted(float(max(v, 0.0)) for v in pole_floor_values)
    orders = sorted(int(v) for v in zolo_r_values)
    for r1 in orders:
        for floor1 in floors:
            step1, _coeffs1, metrics1 = tempered_zolo_candidate(
                ell=ell0, r=r1, pole_floor=floor1, dps=zolo_coeff_dps
            )
            cond1, a1, cancel1 = metrics1
            if cond1 > float(max_cond_allowed) or a1 > float(max_a_allowed):
                continue
            if cancel1 > float(max_cancel_allowed):
                continue
            if require_monotone_gain and step1.ell_out <= ell0:
                continue
            for r2 in orders:
                for floor2 in floors:
                    step2, _coeffs2, metrics2 = tempered_zolo_candidate(
                        ell=step1.ell_out,
                        r=r2,
                        pole_floor=floor2,
                        dps=zolo_coeff_dps,
                    )
                    cond2, a2, cancel2 = metrics2
                    if cond2 > float(max_cond_allowed) or a2 > float(max_a_allowed):
                        continue
                    if cancel2 > float(max_cancel_allowed):
                        continue
                    if require_monotone_gain and step2.ell_out <= step1.ell_out:
                        continue
                    if step2.pred_kappa_after > effective_target:
                        continue
                    key = (
                        float(r1 + r2),
                        float(max(cond1, cond2)),
                        float(max(a1, a2)),
                        float(max(cancel1, cancel2)),
                        float(step2.pred_kappa_after),
                    )
                    if best_key is None or key < best_key:
                        best_key = key
                        best_sched = [step1, step2]

    if best_key is None or best_sched is None:
        return ScheduleInfo(
            "two_step_tempered_cholesky",
            (float("inf"),) * 5,
            [],
        )
    return ScheduleInfo("two_step_tempered_cholesky", best_key, best_sched)
