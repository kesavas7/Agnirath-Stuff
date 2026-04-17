import pandas as pd
import numpy as np
df = pd.read_csv("telemetry.csv")
print("Shape:", df.shape)
print("\nDtypes:\n", df.dtypes)
print("\nNull counts:\n", df.isnull().sum())


df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df.loc[df['velocity_kmh'] < 0, 'velocity_kmh'] = np.nan
df.loc[df['battery_voltage'] > 160, 'battery_voltage'] = np.nan
df.loc[df['solar_irradiance_wm2'] > 1200, 'solar_irradiance_wm2'] = np.nan
df = df.sort_values('timestamp')
df = df.interpolate(method='linear')
df = df.dropna()

panel_efficiency = 0.22
panel_area = 4.0  # m^2
df['power_input'] = (
    df['solar_irradiance_wm2'] *
    panel_efficiency *
    panel_area
)
