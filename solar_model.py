"""
solar_model.py
==============
Phase 1 - The Cartographer: Solar Model
Agnirath Strategy Module | Sasol Solar Challenge Day 2

Models incident solar power on the car's panel throughout race day
using a Gaussian irradiance curve.

Constants (from Agnirath orientation session):
    Panel Area       : 6.0 m²
    Panel Efficiency : 24% (0.24)

Gaussian Model:
    G(t) = G_peak * exp( -(t - t_noon)^2 / (2 * sigma^2) )

    where:
        G_peak  = 1073 W/m²   (peak irradiance at solar noon)
        t_noon  = 12:00 PM    (43200 seconds from midnight)
        sigma   = 11600 s     (standard deviation of daylight curve)

    Actual power delivered to the electrical system:
        P_solar(t) = G(t) * eta_panel * A_panel
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
G_PEAK       = 1073.0   # W/m²  — peak irradiance at solar noon
T_NOON       = 12 * 3600  # seconds from midnight — 12:00 PM = 43200 s
SIGMA        = 11600.0  # seconds — std deviation of Gaussian daylight curve
ETA_PANEL    = 0.24     # panel efficiency (24%)
A_PANEL      = 6.0      # panel area in m²

RACE_START   = 8  * 3600   # 8:00 AM in seconds from midnight
RACE_END     = 17 * 3600   # 5:00 PM in seconds from midnight


# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def solar_irradiance(t_seconds):
    """
    Compute incident solar irradiance at time t (seconds from midnight).
    Uses a Gaussian curve centred at solar noon.

    Parameters
    ----------
    t_seconds : float or np.ndarray
        Time in seconds from midnight.

    Returns
    -------
    G : float or np.ndarray
        Irradiance in W/m². Clipped to 0 (no negative irradiance at night).
    """
    G = G_PEAK * np.exp(-((t_seconds - T_NOON) ** 2) / (2 * SIGMA ** 2))
    return np.maximum(G, 0.0)   # clip negatives (shouldn't occur but safe)


def solar_power(t_seconds):
    """
    Compute actual electrical power delivered by the solar panel at time t.

    P_solar(t) = G(t) * eta_panel * A_panel

    Parameters
    ----------
    t_seconds : float or np.ndarray
        Time in seconds from midnight.

    Returns
    -------
    P : float or np.ndarray
        Solar power in Watts.
    """
    return solar_irradiance(t_seconds) * ETA_PANEL * A_PANEL


def seconds_to_hhmm(t_seconds):
    """Convert seconds-from-midnight to HH:MM string."""
    h = int(t_seconds // 3600)
    m = int((t_seconds % 3600) // 60)
    return f"{h:02d}:{m:02d}"


def energy_during_race():
    """
    Integrate solar power over the full race window (8 AM to 5 PM)
    to get total solar energy available in kWh.
    """
    t = np.linspace(RACE_START, RACE_END, 100_000)
    P = solar_power(t)
    dt = t[1] - t[0]
    E_joules = np.trapezoid(P, t)
    E_kwh    = E_joules / 3_600_000
    return E_kwh


# ─────────────────────────────────────────────────────────────────────────────
# PRINT KEY VALUES
# ─────────────────────────────────────────────────────────────────────────────
def print_summary():
    print("=" * 55)
    print("  AGNIRATH — Solar Model Summary")
    print("=" * 55)
    print(f"\nPanel constants:")
    print(f"  Area             : {A_PANEL} m²")
    print(f"  Efficiency       : {ETA_PANEL*100:.0f}%")
    print(f"  Peak irradiance  : {G_PEAK} W/m²")
    print(f"  Solar noon       : {seconds_to_hhmm(T_NOON)}")
    print(f"  Sigma            : {SIGMA} s  ({SIGMA/3600:.2f} h)")

    print(f"\nKey irradiance values:")
    for label, t in [("08:00 AM (race start)", RACE_START),
                     ("10:00 AM",              10*3600),
                     ("12:00 PM (solar noon)", T_NOON),
                     ("02:00 PM",              14*3600),
                     ("05:00 PM (race end)",   RACE_END)]:
        G = solar_irradiance(t)
        P = solar_power(t)
        print(f"  {label:<28} G = {G:>7.2f} W/m²   P = {P:>6.2f} W")

    E = energy_during_race()
    print(f"\nTotal solar energy (8AM–5PM): {E:.4f} kWh")
    print(f"                             = {E*1000:.1f} Wh")
    print("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────────────────
def plot_solar_model():
    # Full day curve (4 AM to 8 PM)
    t_full   = np.linspace(4*3600, 20*3600, 10_000)
    G_full   = solar_irradiance(t_full)
    P_full   = solar_power(t_full)

    # Race window only
    t_race   = np.linspace(RACE_START, RACE_END, 5_000)
    G_race   = solar_irradiance(t_race)
    P_race   = solar_power(t_race)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                    facecolor='#0d0d0d', sharex=True)
    fig.suptitle('Agnirath Solar Model — Gaussian Irradiance Curve',
                 color='white', fontsize=14, fontweight='bold', y=0.98)
    fig.subplots_adjust(hspace=0.1)

    for ax in [ax1, ax2]:
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='#aaaaaa', labelsize=9)
        ax.xaxis.label.set_color('#aaaaaa')
        ax.yaxis.label.set_color('#aaaaaa')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')

    # ── Irradiance plot ──
    ax1.plot(t_full/3600, G_full, color='#ffcc00', linewidth=2,
             label='G(t) — Irradiance')
    ax1.fill_between(t_full/3600, G_full,
                     where=((t_full >= RACE_START) & (t_full <= RACE_END)),
                     color='#ffcc00', alpha=0.15, label='Race window')
    ax1.axvline(T_NOON/3600, color='#ff8c00', linewidth=1.5,
                linestyle='--', label=f'Solar noon (12:00 PM)\nG = {G_PEAK} W/m²')
    ax1.axvline(RACE_START/3600, color='#ffffff', linewidth=1,
                linestyle=':', alpha=0.6)
    ax1.axvline(RACE_END/3600,   color='#ffffff', linewidth=1,
                linestyle=':', alpha=0.6)
    ax1.set_ylabel('Irradiance (W/m²)')
    ax1.set_title('Incident Solar Irradiance  G(t) = G_peak · exp(-(t-t_noon)² / 2σ²)')
    ax1.legend(fontsize=8, facecolor='#1a1a1a', labelcolor='white',
               edgecolor='#333333', loc='upper right')
    ax1.set_ylim(0, 1200)

    # Annotate peak
    ax1.annotate(f'Peak: {G_PEAK} W/m²',
                 xy=(12, G_PEAK), xytext=(13.5, 950),
                 color='#ffcc00', fontsize=8,
                 arrowprops=dict(arrowstyle='->', color='#ffcc00', lw=1.0))

    # ── Solar power plot ──
    ax2.plot(t_full/3600, P_full, color='#ff8c00', linewidth=2,
             label=f'P_solar(t) = G(t) × {ETA_PANEL} × {A_PANEL} m²')
    ax2.fill_between(t_full/3600, P_full,
                     where=((t_full >= RACE_START) & (t_full <= RACE_END)),
                     color='#ff8c00', alpha=0.15, label='Race window')
    ax2.axvline(T_NOON/3600, color='#ff4444', linewidth=1.5,
                linestyle='--',
                label=f'Peak power = {G_PEAK*ETA_PANEL*A_PANEL:.1f} W')
    ax2.axvline(RACE_START/3600, color='#ffffff', linewidth=1,
                linestyle=':', alpha=0.6, label='Race start/end (8AM–5PM)')
    ax2.axvline(RACE_END/3600,   color='#ffffff', linewidth=1,
                linestyle=':', alpha=0.6)

    # Shade total energy area
    ax2.fill_between(t_race/3600, P_race,
                     color='#ff8c00', alpha=0.08)

    E = energy_during_race()
    ax2.text(10.5, 700,
             f'Total race energy\n= {E:.3f} kWh',
             color='#ff8c00', fontsize=8.5,
             bbox=dict(boxstyle='round', facecolor='#1a1a1a',
                       edgecolor='#ff8c00', alpha=0.8))

    ax2.set_xlabel('Time of Day (hours)')
    ax2.set_ylabel('Solar Power (W)')
    ax2.set_title(f'Actual Panel Power  (η = {ETA_PANEL*100:.0f}%,  A = {A_PANEL} m²)')
    ax2.legend(fontsize=8, facecolor='#1a1a1a', labelcolor='white',
               edgecolor='#333333', loc='upper right')
    ax2.set_ylim(0, 1800)

    # X-axis: show HH:MM labels
    ax2.set_xticks(range(4, 21))
    ax2.set_xticklabels([f"{h:02d}:00" for h in range(4, 21)],
                        rotation=30, ha='right', color='#aaaaaa', fontsize=8)

    plt.savefig('solar_model.png', dpi=150,
                bbox_inches='tight', facecolor='#0d0d0d')
    print("Plot saved as solar_model.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print_summary()
    plot_solar_model()
