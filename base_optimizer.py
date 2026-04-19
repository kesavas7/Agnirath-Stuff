"""
base_optimizer.py
=================
Phase 2 - Base Route Optimizer
Agnirath Strategy Module | Sasol Solar Challenge Day 2

Optimizes the velocity profile from Sasolburg to Zeerust.

Objective : Minimize total travel time (arrive at Zeerust ASAP
            to maximize time available for loops)
Variables : velocity at each route segment (m/s)
Constraints:
    - SOC must stay >= 20% at all times
    - Velocity within [60, 130] km/h
    - Must arrive before 5:00 PM (hard deadline)

Solver: SciPy SLSQP (Sequential Least Squares Programming)
        Handles nonlinear constraints and bounds efficiently.
"""

import numpy as np
import pandas as pd
import time
from scipy.optimize import minimize
from physics import (
    BATTERY_KWH, SOC_MIN, SOC_START, V_MIN_MS, V_MAX_MS,
    energy_segment, solar_power, battery_power, mechanical_power
)

# ─────────────────────────────────────────────────────────────────────────────
# RACE TIMING
# ─────────────────────────────────────────────────────────────────────────────
RACE_START_SEC  = 8 * 3600    # 8:00 AM
RACE_END_SEC    = 17 * 3600   # 5:00 PM
MAX_RACE_TIME_S = RACE_END_SEC - RACE_START_SEC   # 32400 s = 9 hours
CONTROL_STOP_S  = 30 * 60     # mandatory 30 min stop at Zeerust


# ─────────────────────────────────────────────────────────────────────────────
# LOAD ROUTE DATA
# ─────────────────────────────────────────────────────────────────────────────
def load_route(csv_path="route_data.csv"):
    df = pd.read_csv(csv_path)
    slopes = df["slope_deg"].values.copy()
    dists  = df["segment_length_m"].values.copy()
    dists[0] = dists[1]   # fix zero at first point
    return slopes, dists


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION — given a velocity profile, compute time + SOC trajectory
# ─────────────────────────────────────────────────────────────────────────────
def simulate(velocities_ms, slopes, dists):
    """
    Forward-simulate the route given a velocity at each segment.

    Returns
    -------
    times_s   : cumulative time at each point (seconds from race start)
    soc       : SOC at each point (fraction 0-1)
    """
    n         = len(velocities_ms)
    times_s   = np.zeros(n)
    soc       = np.zeros(n)
    soc[0]    = SOC_START
    E_batt_wh = SOC_START * BATTERY_KWH * 1000   # Wh

    for i in range(1, n):
        v     = velocities_ms[i]
        dt    = dists[i] / v                       # time for this segment (s)
        t_abs = RACE_START_SEC + times_s[i-1]      # absolute time (s from midnight)

        dE    = energy_segment(v, slopes[i], dists[i], t_abs)  # Wh
        E_batt_wh  -= dE
        E_batt_wh   = max(E_batt_wh, 0.0)          # clamp

        times_s[i]  = times_s[i-1] + dt
        soc[i]      = E_batt_wh / (BATTERY_KWH * 1000)

    return times_s, soc


# ─────────────────────────────────────────────────────────────────────────────
# OBJECTIVE — minimize total travel time
# ─────────────────────────────────────────────────────────────────────────────
def objective(v, dists):
    """Total travel time in seconds. We minimize this."""
    return np.sum(dists / v)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRAINTS
# ─────────────────────────────────────────────────────────────────────────────
def build_constraints(slopes, dists):
    n = len(slopes)

    def soc_constraint(v):
        """SOC must stay >= SOC_MIN at every point. Returns array >= 0."""
        _, soc = simulate(v, slopes, dists)
        return soc - SOC_MIN   # all values must be >= 0

    def time_constraint(v):
        """Must arrive before 5PM minus control stop time."""
        total_time = np.sum(dists / v)
        max_allowed = MAX_RACE_TIME_S - CONTROL_STOP_S
        return max_allowed - total_time   # must be >= 0

    return [
        {"type": "ineq", "fun": soc_constraint},
        {"type": "ineq", "fun": time_constraint},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# SMART INITIAL GUESS
# ─────────────────────────────────────────────────────────────────────────────
def initial_velocity_profile(slopes, dists):
    """
    Start with a physics-aware initial guess:
    - Uphill   : slower (save energy)
    - Downhill : faster (use gravity)
    - Flat     : cruise at 90 km/h
    Clipped to [V_MIN_MS, V_MAX_MS].
    """
    base = 90 / 3.6   # 90 km/h base
    v0   = np.full(len(slopes), base)

    for i, s in enumerate(slopes):
        if s > 1.0:      # uphill
            v0[i] = max(V_MIN_MS, base - s * 0.8)
        elif s < -1.0:   # downhill
            v0[i] = min(V_MAX_MS, base + abs(s) * 0.8)

    return np.clip(v0, V_MIN_MS, V_MAX_MS)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────
def optimize_base_route(csv_path="route_data.csv", verbose=True):
    slopes, dists = load_route(csv_path)
    n             = len(slopes)

    if verbose:
        print("=" * 60)
        print("  BASE ROUTE OPTIMIZER — Sasolburg → Zeerust")
        print("=" * 60)
        print(f"  Route points   : {n}")
        print(f"  Total distance : {dists.sum()/1000:.2f} km")
        print(f"  Solver         : SLSQP")
        print(f"  Objective      : Minimize travel time")

    # Initial guess
    v0     = initial_velocity_profile(slopes, dists)
    bounds = [(V_MIN_MS, V_MAX_MS)] * n
    cons   = build_constraints(slopes, dists)

    t_start = time.time()

    if verbose:
        print("\n  Optimizing... (this may take a few minutes)")

    result = minimize(
        objective,
        v0,
        args=(dists,),
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={
            "maxiter" : 500,
            "ftol"    : 1e-6,
            "disp"    : verbose,
        },
    )

    elapsed = time.time() - t_start

    if verbose:
        print(f"\n  Converged      : {result.success}")
        print(f"  Message        : {result.message}")
        print(f"  Time taken     : {elapsed:.1f} s")

    # ── Post-process results ──
    v_opt          = result.x
    times_s, soc   = simulate(v_opt, slopes, dists)

    total_time_s   = times_s[-1]
    arrival_sec    = RACE_START_SEC + total_time_s
    arrival_h      = int(arrival_sec // 3600)
    arrival_m      = int((arrival_sec % 3600) // 60)
    time_for_loops = RACE_END_SEC - arrival_sec - CONTROL_STOP_S

    if verbose:
        print(f"\n  Results:")
        print(f"  Travel time    : {total_time_s/3600:.3f} h  "
              f"({total_time_s/60:.1f} min)")
        print(f"  Arrival time   : {arrival_h:02d}:{arrival_m:02d}")
        print(f"  SOC at arrival : {soc[-1]*100:.1f}%")
        print(f"  Time for loops : {time_for_loops/3600:.3f} h  "
              f"({time_for_loops/60:.1f} min)")
        print(f"  Min SOC en route: {soc.min()*100:.1f}%")
        print("=" * 60)

    return {
        "velocities_ms"  : v_opt,
        "times_s"        : times_s,
        "soc"            : soc,
        "slopes"         : slopes,
        "dists"          : dists,
        "total_time_s"   : total_time_s,
        "arrival_sec"    : arrival_sec,
        "time_for_loops_s": time_for_loops,
        "soc_at_arrival" : soc[-1],
        "converged"      : result.success,
    }


if __name__ == "__main__":
    res = optimize_base_route()
