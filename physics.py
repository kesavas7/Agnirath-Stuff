"""
physics.py
==========
Core physics model for Agnirath solar car.
All power values in Watts, energy in kWh, velocity in m/s.

Car Constants:
    Mass            : 250 kg
    CdA             : 0.118 m²
    Crr             : 0.004
    Battery         : 3.1 kWh
    Panel area      : 6.0 m²
    Panel efficiency: 24%
    Motor efficiency: 90%
    Regen efficiency: 70%
"""

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CAR CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
MASS          = 250.0    # kg
CDA           = 0.118    # m²  (drag coefficient × frontal area)
CRR           = 0.004    # rolling resistance coefficient
RHO_AIR       = 1.2      # kg/m³ air density
G             = 9.81     # m/s²

BATTERY_KWH   = 3.1      # kWh total capacity
BATTERY_J     = BATTERY_KWH * 3_600_000  # joules

ETA_MOTOR     = 0.90     # motor efficiency
ETA_REGEN     = 0.70     # regenerative braking efficiency

PANEL_AREA    = 6.0      # m²
PANEL_EFF     = 0.24     # 24%

# Solar Gaussian model constants
G_PEAK        = 1073.0   # W/m²
T_NOON        = 43200.0  # seconds from midnight (12:00 PM)
SIGMA         = 11600.0  # seconds

# Speed limits
V_MIN_MS      = 60  / 3.6   # 60  km/h in m/s
V_MAX_MS      = 130 / 3.6   # 130 km/h in m/s

# SOC limits
SOC_MIN       = 0.20     # never drop below 20%
SOC_START     = 1.00     # start Day 2 fully charged


# ─────────────────────────────────────────────────────────────────────────────
# SOLAR MODEL
# ─────────────────────────────────────────────────────────────────────────────
def solar_irradiance(t_sec):
    """Gaussian irradiance W/m² at time t (seconds from midnight)."""
    G = G_PEAK * np.exp(-((t_sec - T_NOON) ** 2) / (2 * SIGMA ** 2))
    return np.maximum(G, 0.0)


def solar_power(t_sec):
    """Actual panel power (W) at time t (seconds from midnight)."""
    return solar_irradiance(t_sec) * PANEL_EFF * PANEL_AREA


# ─────────────────────────────────────────────────────────────────────────────
# FORCE & POWER MODEL
# ─────────────────────────────────────────────────────────────────────────────
def aero_drag_power(v_ms):
    """Aerodynamic drag power (W). P_aero = 0.5 * rho * CdA * v^3"""
    return 0.5 * RHO_AIR * CDA * v_ms ** 3


def rolling_resistance_power(v_ms, slope_deg=0.0):
    """Rolling resistance power (W). P_rr = Crr * m * g * cos(theta) * v"""
    theta = np.radians(slope_deg)
    return CRR * MASS * G * np.cos(theta) * v_ms


def gravity_power(v_ms, slope_deg):
    """Gravitational power (W). Positive = uphill (costs energy)."""
    theta = np.radians(slope_deg)
    return MASS * G * np.sin(theta) * v_ms


def mechanical_power(v_ms, slope_deg=0.0):
    """
    Total mechanical power required at wheels (W).
    Positive = motoring (draining battery).
    Negative = car would naturally accelerate (regen territory).
    """
    return (aero_drag_power(v_ms)
            + rolling_resistance_power(v_ms, slope_deg)
            + gravity_power(v_ms, slope_deg))


def battery_power(v_ms, slope_deg=0.0):
    """
    Net electrical power drawn from battery (W).
    Positive = draining. Negative = charging (regen).
    Accounts for motor and regen efficiencies.
    """
    P_mech = mechanical_power(v_ms, slope_deg)
    if np.isscalar(P_mech):
        if P_mech >= 0:
            return P_mech / ETA_MOTOR
        else:
            return P_mech * ETA_REGEN
    else:
        P_batt = np.where(P_mech >= 0,
                          P_mech / ETA_MOTOR,
                          P_mech * ETA_REGEN)
        return P_batt


def net_power(v_ms, slope_deg, t_sec):
    """
    Net power balance (W).
    Positive = battery draining.
    Negative = battery charging (solar exceeds draw).
    """
    return battery_power(v_ms, slope_deg) - solar_power(t_sec)


def energy_segment(v_ms, slope_deg, dist_m, t_sec):
    """
    Energy drawn from battery (Wh) to traverse a segment.

    Parameters
    ----------
    v_ms      : velocity in m/s
    slope_deg : slope in degrees
    dist_m    : segment distance in metres
    t_sec     : time at start of segment (seconds from midnight)

    Returns
    -------
    delta_E_wh : energy drawn (Wh). Positive = drain. Negative = gain.
    """
    time_s     = dist_m / v_ms                  # time to cross segment
    P_batt     = battery_power(v_ms, slope_deg)
    P_sol      = solar_power(t_sec + time_s / 2) # solar at midpoint time
    P_net      = P_batt - P_sol
    return P_net * time_s / 3600                 # W*s -> Wh
