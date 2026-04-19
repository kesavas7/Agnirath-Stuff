"""
plot_results.py
===============
Phase 3 - The Analyst: Outputs & Visualization
Agnirath Strategy Module | Sasol Solar Challenge Day 2

Generates all required plots:
    1. Velocity Profile (vs time)
    2. State of Charge Profile (vs time)
    3. Acceleration Profile (vs time)
    4. Solar Power Profile (vs time)
    5. Elevation Profile (vs distance)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from physics import solar_power, BATTERY_KWH
from loop_optimizer import LOOP_DIST_M, STOP_BETWEEN_S

RACE_START_SEC = 8 * 3600
RACE_END_SEC   = 17 * 3600


def seconds_to_hhmm(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    return f"{h:02d}:{m:02d}"


def build_full_timeline(base_res, loop_res, slopes, dists):
    """
    Stitch together the full day timeline:
    base route + control stop + loops + inter-loop stops.

    Returns arrays of (time_sec, velocity_ms, soc, solar_W)
    sampled at 30-second intervals.
    """
    dt         = 30   # seconds per sample
    t_abs_arr  = []
    v_arr      = []
    soc_arr    = []
    sol_arr    = []

    # ── Base route ──────────────────────────────────────────
    v_base   = base_res["velocities_ms"]
    t_base   = base_res["times_s"]
    soc_base = base_res["soc"]

    # Interpolate to uniform 30-s grid
    t_base_abs = RACE_START_SEC + t_base
    t_grid_base = np.arange(RACE_START_SEC,
                             RACE_START_SEC + t_base[-1], dt)
    v_interp    = np.interp(t_grid_base, t_base_abs, v_base)
    soc_interp  = np.interp(t_grid_base, t_base_abs, soc_base)

    t_abs_arr.extend(t_grid_base.tolist())
    v_arr.extend(v_interp.tolist())
    soc_arr.extend(soc_interp.tolist())
    sol_arr.extend(solar_power(t_grid_base).tolist())

    # ── Control stop (30 min, v = 0) ────────────────────────
    t_arrival   = RACE_START_SEC + t_base[-1]
    t_loop_start = loop_res["t_loop_start"]
    t_stop_grid  = np.arange(t_arrival, t_loop_start, dt)
    soc_stop     = soc_base[-1]

    t_abs_arr.extend(t_stop_grid.tolist())
    v_arr.extend([0.0] * len(t_stop_grid))
    soc_arr.extend([soc_stop] * len(t_stop_grid))
    sol_arr.extend(solar_power(t_stop_grid).tolist())

    # ── Loops ───────────────────────────────────────────────
    N       = loop_res["N_loops"]
    v_loop  = loop_res["v_optimal_ms"]
    t_cur   = loop_res["t_loop_start"]
    soc_cur = loop_res["soc_loop_start"]

    for loop_idx in range(N):
        loop_time_s = LOOP_DIST_M / v_loop
        t_end_loop  = t_cur + loop_time_s

        # Driving the loop
        t_loop_grid = np.arange(t_cur, t_end_loop, dt)
        dE_per_s    = (solar_power(t_loop_grid) - 0) / 3600  # approximate
        # SOC during loop: linear drain
        E_start_wh  = soc_cur * BATTERY_KWH * 1000
        from loop_optimizer import loop_energy_wh
        dE_loop = loop_energy_wh(v_loop, t_cur)   # Wh for full loop
        soc_end_loop = soc_cur - dE_loop / (BATTERY_KWH * 1000)

        soc_loop_interp = np.linspace(soc_cur, soc_end_loop, len(t_loop_grid))

        t_abs_arr.extend(t_loop_grid.tolist())
        v_arr.extend([v_loop] * len(t_loop_grid))
        soc_arr.extend(soc_loop_interp.tolist())
        sol_arr.extend(solar_power(t_loop_grid).tolist())

        soc_cur = soc_end_loop
        t_cur   = t_end_loop

        # Inter-loop stop (except after last loop)
        if loop_idx < N - 1:
            t_stop_end  = t_cur + STOP_BETWEEN_S
            t_stop_grid = np.arange(t_cur, t_stop_end, dt)
            t_abs_arr.extend(t_stop_grid.tolist())
            v_arr.extend([0.0] * len(t_stop_grid))
            soc_arr.extend([soc_cur] * len(t_stop_grid))
            sol_arr.extend(solar_power(t_stop_grid).tolist())
            t_cur = t_stop_end

    return (np.array(t_abs_arr), np.array(v_arr),
            np.array(soc_arr),   np.array(sol_arr))


def compute_acceleration(v_arr, t_arr):
    """Acceleration in m/s² from velocity array."""
    acc = np.gradient(v_arr, t_arr)
    return acc


def plot_all(base_res, loop_res, slopes, dists, save_path="full_day_strategy.png"):
    t_abs, v_ms, soc, sol = build_full_timeline(
        base_res, loop_res, slopes, dists)

    t_hours = t_abs / 3600
    v_kmh   = v_ms * 3.6
    acc     = compute_acceleration(v_ms, t_abs)

    # ── Key event times ──────────────────────────────────────
    t_arrival_h   = (RACE_START_SEC + base_res["total_time_s"]) / 3600
    t_loop_h      = loop_res["t_loop_start"] / 3600
    t_final_h     = loop_res["t_final_sec"] / 3600 if loop_res["t_final_sec"] else 17.0

    # ── Figure ───────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 14), facecolor='#0d0d0d')
    fig.suptitle(
        'Agnirath — Day 2 Full Race Strategy\n'
        'Sasolburg → Zeerust + Distance Maximization Loops',
        color='white', fontsize=14, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, :])   # velocity — full width
    ax2 = fig.add_subplot(gs[1, :])   # SOC — full width
    ax3 = fig.add_subplot(gs[2, 0])   # acceleration
    ax4 = fig.add_subplot(gs[2, 1])   # solar power

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='#aaaaaa', labelsize=9)
        ax.xaxis.label.set_color('#aaaaaa')
        ax.yaxis.label.set_color('#aaaaaa')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')

    def add_event_lines(ax, y_text_pos):
        ax.axvline(t_arrival_h, color='#00ccff', lw=1.2, ls='--', alpha=0.8)
        ax.axvline(t_loop_h,    color='#ffcc00', lw=1.2, ls='--', alpha=0.8)
        ax.axvline(17.0,        color='#ff4444', lw=1.2, ls='--', alpha=0.8)
        ax.text(t_arrival_h + 0.03, y_text_pos, 'Arrival\nZeerust',
                color='#00ccff', fontsize=7, va='top')
        ax.text(t_loop_h + 0.03,    y_text_pos, 'Loops\nStart',
                color='#ffcc00', fontsize=7, va='top')
        ax.text(17.03,              y_text_pos, '5PM\nEnd',
                color='#ff4444', fontsize=7, va='top')

    # ── Plot 1: Velocity ─────────────────────────────────────
    ax1.plot(t_hours, v_kmh, color='#ff8c00', linewidth=1.5, label='Velocity')
    ax1.fill_between(t_hours, v_kmh, alpha=0.08, color='#ff8c00')
    ax1.axhline(60,  color='#ff4444', lw=0.8, ls=':', alpha=0.6, label='v_min = 60 km/h')
    ax1.axhline(130, color='#ff4444', lw=0.8, ls=':', alpha=0.6, label='v_max = 130 km/h')
    add_event_lines(ax1, 125)
    ax1.set_xlabel('Time of Day (h)')
    ax1.set_ylabel('Velocity (km/h)')
    ax1.set_title('Velocity Profile  (v = 0 during control stops)')
    ax1.set_xlim(8, 17.5)
    ax1.set_ylim(-5, 145)
    ax1.legend(fontsize=8, facecolor='#1a1a1a', labelcolor='white',
               edgecolor='#333', loc='upper right')
    ax1.set_xticks(np.arange(8, 18))
    ax1.set_xticklabels([f"{h:02d}:00" for h in range(8, 18)], color='#aaaaaa')

    # ── Plot 2: SOC ──────────────────────────────────────────
    ax2.plot(t_hours, soc * 100, color='#00ff88', linewidth=1.5, label='Battery SOC')
    ax2.fill_between(t_hours, soc * 100, 20, where=soc * 100 >= 20,
                     color='#00ff88', alpha=0.08)
    ax2.axhline(20, color='#ff4444', lw=1.2, ls='--', label='SOC minimum (20%)')
    ax2.axhline(100, color='#aaaaaa', lw=0.8, ls=':', alpha=0.5)
    add_event_lines(ax2, 95)
    ax2.set_xlabel('Time of Day (h)')
    ax2.set_ylabel('State of Charge (%)')
    ax2.set_title('Battery SOC Profile')
    ax2.set_xlim(8, 17.5)
    ax2.set_ylim(0, 110)
    ax2.legend(fontsize=8, facecolor='#1a1a1a', labelcolor='white',
               edgecolor='#333', loc='upper right')
    ax2.set_xticks(np.arange(8, 18))
    ax2.set_xticklabels([f"{h:02d}:00" for h in range(8, 18)], color='#aaaaaa')

    # ── Plot 3: Acceleration ─────────────────────────────────
    ax3.plot(t_hours, acc, color='#ff4488', linewidth=0.8, alpha=0.85)
    ax3.axhline(0, color='#ffffff', lw=0.7, ls='-', alpha=0.3)
    ax3.axhline( 1.5, color='#ff4444', lw=0.8, ls=':', alpha=0.6,
                label='±1.5 m/s² limit')
    ax3.axhline(-1.5, color='#ff4444', lw=0.8, ls=':', alpha=0.6)
    ax3.set_xlabel('Time of Day (h)')
    ax3.set_ylabel('Acceleration (m/s²)')
    ax3.set_title('Acceleration Profile')
    ax3.set_xlim(8, 17.5)
    ax3.set_ylim(-3, 3)
    ax3.legend(fontsize=8, facecolor='#1a1a1a', labelcolor='white', edgecolor='#333')
    ax3.set_xticks(np.arange(8, 18, 2))
    ax3.set_xticklabels([f"{h:02d}:00" for h in range(8, 18, 2)], color='#aaaaaa')

    # ── Plot 4: Solar Power ──────────────────────────────────
    ax4.plot(t_hours, sol, color='#ffcc00', linewidth=1.5, label='Solar power')
    ax4.fill_between(t_hours, sol, alpha=0.12, color='#ffcc00')
    ax4.set_xlabel('Time of Day (h)')
    ax4.set_ylabel('Solar Power (W)')
    ax4.set_title('Solar Panel Power (Gaussian Model)')
    ax4.set_xlim(8, 17.5)
    ax4.set_ylim(0, 1700)
    ax4.legend(fontsize=8, facecolor='#1a1a1a', labelcolor='white', edgecolor='#333')
    ax4.set_xticks(np.arange(8, 18, 2))
    ax4.set_xticklabels([f"{h:02d}:00" for h in range(8, 18, 2)], color='#aaaaaa')
    ax4.text(11.8, 1580, f'Peak: 1545 W\n@ 12:00 PM',
             color='#ffcc00', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='#1a1a1a',
                       edgecolor='#ffcc00', alpha=0.8))

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
    print(f"[Plot] Saved to {save_path}")
    return fig
