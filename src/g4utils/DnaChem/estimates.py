"""Explicit particle-stage G estimators from ScoreSpecies aggregate rows."""

from __future__ import annotations

import math
from numbers import Integral, Real
from pathlib import Path

import pandas as pd


# Geant4 wcsv prints rounded decimal moments. Allow two parts in 100,000
# when checking the nonnegative centered sum of
# squares; the ulp term also covers subtraction roundoff after CSV parsing.
MOMENT_REL_TOL = 2e-5


def validate_total_energy(value: object, manifest_path: Path) -> float:
    """Require a usable Dump-wide energy denominator before any division."""
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value <= 0):
        raise ValueError(
            f"{manifest_path}: totalEnergyDeposit_eV must be finite and positive "
            "for particle G estimates"
        )
    return float(value)


def particle_g_estimates(
    frame: pd.DataFrame,
    total_energy_eV: float,
    source: Path,
    *,
    assume_valid_event_g: bool = False,
) -> pd.DataFrame:
    """Add the two G estimators and an optional two-moment SEM to raw rows.

    ``assume_valid_event_g=True`` asserts that *both* moments include exactly
    ``nEvent`` valid event G values, including zeros, and that each event's
    deposited energy was positive. It does not verify event independence.
    """
    required = {"number", "nEvent", "speciesName", "time", "sumG", "sumG2"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source}: missing required species columns: {missing}")
    ratios: list[float] = []
    means: list[float] = []
    sems: list[float | None] = []
    for pos in range(len(frame)):
        row = frame.iloc[pos]
        context = f"{source}: row {pos + 1} ({row['speciesName']!r}, time={row['time']} ns)"
        n_event = frame["nEvent"].iloc[pos]
        if (isinstance(n_event, bool) or not isinstance(n_event, Integral)
                or n_event <= 0):
            raise ValueError(f"{context}: nEvent must be a positive integer")
        n_event = int(n_event)  # Avoid NumPy int64 overflow in the SEM denominator.
        sum_g = frame["sumG"].iloc[pos]
        sum_g2 = frame["sumG2"].iloc[pos]
        if not isinstance(sum_g, Real) or not math.isfinite(sum_g):
            raise ValueError(f"{context}: sumG must be finite")
        if (not isinstance(sum_g2, Real) or not math.isfinite(sum_g2)
                or sum_g2 < 0):
            raise ValueError(f"{context}: sumG2 must be finite and nonnegative")

        # Evaluate sumG²/nEvent without squaring sumG first. A finite input
        # whose square still overflows cannot have a finite consistent sumG2.
        lower_moment = (sum_g / math.sqrt(n_event)) ** 2
        if not math.isfinite(lower_moment):
            raise ValueError(f"{context}: sumG and sumG2 are inconsistent (overflow)")
        scale = max(abs(sum_g2), lower_moment)
        tolerance = MOMENT_REL_TOL * scale + 64 * math.ulp(scale)
        centered = sum_g2 - lower_moment
        if centered < -tolerance or (n_event == 1 and centered > tolerance):
            raise ValueError(
                f"{context}: sumG2 is inconsistent with sumG and nEvent "
                f"(deficit {abs(centered):g}, tolerance {tolerance:g})"
            )

        ratio = float(frame["number"].iloc[pos]) * 100.0 / total_energy_eV
        if not math.isfinite(ratio):
            raise ValueError(f"{context}: aggregate_energy_G is not finite")
        ratios.append(ratio)
        means.append(sum_g / n_event)
        sems.append(
            math.sqrt(max(0.0, centered) / ((n_event - 1) * n_event))
            if assume_valid_event_g and n_event >= 2 else None
        )

    result = frame.copy()
    result["aggregate_energy_G"] = pd.Series(ratios, index=frame.index, dtype="float64")
    result["mean_event_G"] = pd.Series(means, index=frame.index, dtype="float64")
    result["mean_event_G_sem"] = pd.Series(sems, index=frame.index, dtype="Float64")
    return result
