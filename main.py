"""
main.py
=======
Agnirath Strategy Module | Sasol Solar Challenge Day 2
Full pipeline: Route → Physics → Optimize → Visualize

Run this file to execute the complete strategy simulation.
"""

import time
import numpy as np
from base_optimizer import optimize_base_route
from loop_optimizer  import optimize_loops
from plot_results    import plot_all

RACE_START_SEC = 8 * 3600


def main():
    print("\n" + "=" * 60)
    print("  AGNIRATH — DAY 2 FULL STRATEGY PIPELINE")
    print("=" * 60 + "\n")

    total_start = time.time()

    # ── PHASE 2A: Optimize base route ─────────────────────────
    print("PHASE 2A — Base Route Optimizer")
    print("-" * 60)
    base_res = optimize_base_route(csv_path="route_data.csv", verbose=True)

    if not base_res["converged"]:
        print("\n[WARNING] Base optimizer did not fully converge.")
        print("          Results may be suboptimal. Continuing anyway...\n")

    # ── PHASE 2B: Optimize loops ───────────────────────────────
    print("\nPHASE 2B — Loop Optimizer")
    print("-" * 60)
    loop_res = optimize_loops(
        t_arrival_sec = base_res["arrival_sec"],
        soc_arrival   = base_res["soc_at_arrival"],
        verbose       = True
    )

    # ── SUMMARY ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FULL DAY STRATEGY SUMMARY")
    print("=" * 60)

    base_dist  = base_res["dists"].sum() / 1000
    loop_dist  = loop_res["total_loop_km"]
    total_dist = base_dist + loop_dist

    arr_h = int(base_res["arrival_sec"] // 3600)
    arr_m = int((base_res["arrival_sec"] % 3600) // 60)

    fin_sec = loop_res.get("t_final_sec") or base_res["arrival_sec"]
    fin_h   = int(fin_sec // 3600)
    fin_m   = int((fin_sec % 3600) // 60)

    print(f"  Base route distance  : {base_dist:.2f} km")
    print(f"  Arrival at Zeerust   : {arr_h:02d}:{arr_m:02d}")
    print(f"  SOC at arrival       : {base_res['soc_at_arrival']*100:.1f}%")
    print(f"  Control stop         : 30 min")
    print(f"  Loops completed      : {loop_res['N_loops']}")
    print(f"  Loop velocity        : {loop_res['v_optimal_ms']*3.6:.2f} km/h")
    print(f"  Loop distance        : {loop_dist:.1f} km")
    print(f"  Total distance       : {total_dist:.2f} km")
    print(f"  Final SOC            : {loop_res['soc_final']*100:.1f}%")
    print(f"  Finished at          : {fin_h:02d}:{fin_m:02d}")
    print(f"  Total pipeline time  : {time.time()-total_start:.1f} s")
    print("=" * 60)

    # ── PHASE 3: Plot ─────────────────────────────────────────
    print("\nPHASE 3 — Generating Plots")
    print("-" * 60)
    plot_all(
        base_res  = base_res,
        loop_res  = loop_res,
        slopes    = base_res["slopes"],
        dists     = base_res["dists"],
        save_path = "full_day_strategy.png"
    )
    print("\nAll done. Check full_day_strategy.png for results.")


if __name__ == "__main__":
    main()
