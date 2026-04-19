"""
loop_optimizer.py
=================
Phase 2 - Loop Optimizer
Agnirath Strategy Module | Sasol Solar Challenge Day 2

Given time and SOC remaining after arrival at Zeerust,
finds the maximum number of 35 km loops completable
before 5:00 PM with SOC >= 20% at end.

Rules:
    - Each loop = 35 km (straight-line physics, flat terrain)
    - 5-minute mandatory stop between each loop
    - All driving must cease by 5:00 PM
    - Battery SOC must be >= 20% at 5:00 PM
    - Optimal velocity = minimum feasible (minimizes drag, maximizes solar)

FIX: SOC at loop start is now correctly computed AFTER solar charging
     during the 30-minute control stop, not at raw arrival SOC.
"""

import numpy as np
from physics import (
    BATTERY_KWH, SOC_MIN, V_MIN_MS, V_MAX_MS,
    aero_drag_power, rolling_resistance_power,
    solar_power, ETA_MOTOR
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
LOOP_DIST_M    = 35_000    # 35 km per loop in metres
STOP_BETWEEN_S = 5 * 60    # 5-minute mandatory stop between loops
CONTROL_STOP_S = 30 * 60   # 30-minute mandatory control stop at Zeerust
RACE_END_SEC   = 17 * 3600  # 5:00 PM


# ─────────────────────────────────────────────────────────────────────────────
# SOLAR CHARGING DURING A STATIONARY STOP
# ─────────────────────────────────────────────────────────────────────────────
def soc_after_charging(t_start_sec, duration_s, soc_start, dt=30):
    """
    Compute SOC after the car charges from solar panels while stationary.

    Motor draw = 0 during stops, so all solar power goes into the battery.

    Parameters
    ----------
    t_start_sec : absolute start time of stop (s from midnight)
    duration_s  : length of stop in seconds
    soc_start   : SOC fraction at start of stop
    dt          : integration timestep in seconds

    Returns
    -------
    soc_end : SOC fraction at end of stop
    """
    E_batt_wh = soc_start * BATTERY_KWH * 1000

    for t in np.arange(t_start_sec, t_start_sec + duration_s, dt):
        P_solar   = solar_power(t)
        dE        = P_solar * (dt / 3600)                      # Wh gained
        E_batt_wh = min(E_batt_wh + dE, BATTERY_KWH * 1000)   # clamp to 100%

    return E_batt_wh / (BATTERY_KWH * 1000)


# ─────────────────────────────────────────────────────────────────────────────
# LOOP PHYSICS (flat terrain, slope = 0)
# ─────────────────────────────────────────────────────────────────────────────
def loop_battery_power(v_ms):
    """Electrical power drawn from battery while driving a loop (W). Flat terrain."""
    P_mech = aero_drag_power(v_ms) + rolling_resistance_power(v_ms, 0.0)
    return P_mech / ETA_MOTOR


def loop_net_power(v_ms, t_sec_midpoint):
    """Net battery drain during loop at midpoint time (W)."""
    return loop_battery_power(v_ms) - solar_power(t_sec_midpoint)


def loop_energy_wh(v_ms, t_sec_start):
    """
    Net energy drawn from battery (Wh) to complete one loop.
    Solar evaluated at midpoint time of the loop.
    Negative value means solar exceeds motor draw (battery gains energy).
    """
    loop_time_s = LOOP_DIST_M / v_ms
    t_mid       = t_sec_start + loop_time_s / 2
    P_net       = loop_net_power(v_ms, t_mid)
    return P_net * loop_time_s / 3600   # Wh


# ─────────────────────────────────────────────────────────────────────────────
# FEASIBILITY CHECK FOR N LOOPS AT VELOCITY v
# ─────────────────────────────────────────────────────────────────────────────
def check_feasibility(N, v_ms, t_start_sec, soc_start):
    """
    Check if N loops at v_ms are feasible given:
        t_start_sec : absolute time when loop phase begins (after control stop)
        soc_start   : SOC fraction at start of loop phase
                      (should already include control-stop solar charging)

    Returns (feasible, soc_final, t_final_sec)
    """
    E_available_wh = (soc_start - SOC_MIN) * BATTERY_KWH * 1000
    t_current      = t_start_sec
    E_spent_wh     = 0.0

    for loop_idx in range(N):
        loop_time_s = LOOP_DIST_M / v_ms

        # Time check: must finish loop before 5 PM
        if t_current + loop_time_s > RACE_END_SEC:
            return False, None, None

        # Energy for this loop (can be negative = solar tops up battery)
        dE          = loop_energy_wh(v_ms, t_current)
        E_spent_wh += dE
        t_current  += loop_time_s

        # Energy budget check
        if E_spent_wh > E_available_wh:
            return False, None, None

        # Inter-loop stop: car charges from solar panels
        if loop_idx < N - 1:
            # Solar charging during the 5-minute stop
            soc_before_stop = soc_start - (E_spent_wh / (BATTERY_KWH * 1000))
            soc_after_stop  = soc_after_charging(t_current, STOP_BETWEEN_S, soc_before_stop)
            # Recalculate E_spent_wh to reflect the charging gain
            E_spent_wh = (soc_start - soc_after_stop) * BATTERY_KWH * 1000

            t_current += STOP_BETWEEN_S
            if t_current >= RACE_END_SEC:
                return False, None, None

    soc_final = soc_start - (E_spent_wh / (BATTERY_KWH * 1000))
    return True, soc_final, t_current


# ─────────────────────────────────────────────────────────────────────────────
# FIND OPTIMAL VELOCITY FOR N LOOPS
# ─────────────────────────────────────────────────────────────────────────────
def find_optimal_velocity(N, t_start_sec, soc_start, n_search=5000):
    """
    For a fixed N, scan velocity range to find:
        v_min : minimum feasible speed (time constraint)
        v_max : maximum feasible speed (energy constraint)
    Returns (v_optimal, v_min, v_max) or (None, None, None) if infeasible.
    """
    v_range    = np.linspace(V_MIN_MS, V_MAX_MS, n_search)
    feasible_v = []

    for v in v_range:
        ok, soc_f, t_f = check_feasibility(N, v, t_start_sec, soc_start)
        if ok:
            feasible_v.append(v)

    if len(feasible_v) == 0:
        return None, None, None

    v_min     = min(feasible_v)
    v_max     = max(feasible_v)
    v_optimal = v_min   # minimum speed → minimum drag + maximum time under solar

    return v_optimal, v_min, v_max


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────
def optimize_loops(t_arrival_sec, soc_arrival, verbose=True):
    """
    Find maximum N and optimal velocity for the loop phase.

    Parameters
    ----------
    t_arrival_sec : absolute time of arrival at Zeerust (s from midnight)
    soc_arrival   : SOC fraction at arrival (BEFORE control stop charging)

    Returns dict with results.
    """
    t_loop_start = t_arrival_sec + CONTROL_STOP_S

    # ── Account for solar charging during the 30-min control stop ──
    soc_loop_start = soc_after_charging(t_arrival_sec, CONTROL_STOP_S, soc_arrival)

    if verbose:
        print("=" * 60)
        print("  LOOP OPTIMIZER — Zeerust Distance Maximization")
        print("=" * 60)
        arr_h = int(t_arrival_sec // 3600)
        arr_m = int((t_arrival_sec % 3600) // 60)
        lp_h  = int(t_loop_start // 3600)
        lp_m  = int((t_loop_start % 3600) // 60)
        print(f"  Arrival time       : {arr_h:02d}:{arr_m:02d}")
        print(f"  SOC at arrival     : {soc_arrival*100:.1f}%")
        print(f"  Solar charging     : +{(soc_loop_start - soc_arrival)*100:.1f}% "
              f"during 30-min stop")
        print(f"  Loop phase starts  : {lp_h:02d}:{lp_m:02d}")
        print(f"  SOC at loop start  : {soc_loop_start*100:.1f}%")
        print(f"  Energy available   : "
              f"{(soc_loop_start - SOC_MIN)*BATTERY_KWH*1000:.1f} Wh above minimum")
        print()
        print(f"  {'N':>3}  {'v_min km/h':>12}  {'v_max km/h':>12}  "
              f"{'Feasible':>10}  {'Optimal km/h':>13}")
        print("  " + "-" * 55)

    best_N    = 0
    best_v    = None
    best_vmin = None
    best_vmax = None

    for N in range(1, 20):
        v_opt, v_min, v_max = find_optimal_velocity(
            N, t_loop_start, soc_loop_start)   # ← use charged SOC

        if v_opt is None:
            if verbose:
                print(f"  {N:>3}  {'—':>12}  {'—':>12}  {'NO':>10}  {'—':>13}")
            break

        if verbose:
            print(f"  {N:>3}  {v_min*3.6:>12.2f}  {v_max*3.6:>12.2f}  "
                  f"{'YES':>10}  {v_opt*3.6:>13.2f}")

        best_N    = N
        best_v    = v_opt
        best_vmin = v_min
        best_vmax = v_max

    # Final simulation with best N and v
    ok, soc_final, t_final = check_feasibility(
        best_N, best_v, t_loop_start, soc_loop_start)

    total_loop_dist = best_N * LOOP_DIST_M / 1000   # km

    if verbose:
        print("  " + "-" * 55)
        print(f"\n  >>> MAX LOOPS         : {best_N}")
        print(f"  >>> OPTIMAL VELOCITY  : {best_v*3.6:.2f} km/h")
        print(f"  >>> TOTAL LOOP DIST   : {total_loop_dist:.1f} km")
        print(f"  >>> FINAL SOC         : {soc_final*100:.1f}%")
        fin_h = int(t_final // 3600)
        fin_m = int((t_final % 3600) // 60)
        print(f"  >>> FINISH TIME       : {fin_h:02d}:{fin_m:02d}")
        print("=" * 60)

    return {
        "N_loops"        : best_N,
        "v_optimal_ms"   : best_v,
        "v_min_ms"       : best_vmin,
        "v_max_ms"       : best_vmax,
        "soc_final"      : soc_final,
        "t_final_sec"    : t_final,
        "total_loop_km"  : total_loop_dist,
        "t_loop_start"   : t_loop_start,
        "soc_loop_start" : soc_loop_start,   # post-charging SOC
        "soc_at_arrival" : soc_arrival,       # raw arrival SOC (for reference)
    }


if __name__ == "__main__":
    # Test with example values
    t_arr = 14 * 3600   # arrive at 2:00 PM
    soc   = 0.65        # 65% SOC at arrival
    optimize_loops(t_arr, soc)
