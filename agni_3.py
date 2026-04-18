import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FILE_PATH = "telemetry_A.csv"   # path to the CSV
MASS = 1500                      # vehicle mass
RHO = 1.225                      # air density 
G = 9.81                         # gravity 
SMOOTH_WINDOW = 5                # rolling average window

df = pd.read_csv(FILE_PATH)

df = df.dropna() #drop all null values

df = df[(df['velocity_ms'] > 0) & (df['velocity_ms'] < 100)]
df = df[(df['Gradient_deg'] > -20) & (df['Gradient_deg'] < 20)]
# above 2 lines drops all non realistic values

df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')

df['time_s'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
#time change is found inorder to calculate acceleration at every velocity change
df['theta'] = np.deg2rad(df['Gradient_deg'])

df['velocity_smooth'] = df['velocity_ms'].rolling(
    window=SMOOTH_WINDOW, center=True
).mean()
#velocity is smoothened out by taking the nearest 5 values of that paricular value and averaging

df = df.dropna()  #null things are dropped again

df['acceleration'] = np.gradient(
    df['velocity_smooth'], df['time_s']
)
#new row acc is created from vel and time_s

v = df['velocity_smooth'].values
a = df['acceleration'].values
theta = df['theta'].values

Y = a + G * np.sin(theta) #y is defined as -k1*v^2 - k2
X = np.vstack([v**2, np.ones_like(v)]).T # creates [v^2, 1] arrays

coeffs, _, _, _ = np.linalg.lstsq(X, -Y, rcond=None)
k1, k2 = coeffs # the value of resistances are obtained by fitting in the curve


CdA = (2 * MASS * k1) / RHO
Crr = k2 / G
#then the value of coefficients are found

print("\nResults:")
print("CdA:", CdA)
print("Crr:", Crr)

a_pred = -(k1 * v**2 + k2) - G * np.sin(theta)
#acceleration is found again from the predicted values to plot the graph

plt.figure(figsize=(8, 5))
plt.scatter(v, a, s=10, label="Actual")
plt.scatter(v, a_pred, s=10, label="Fitted")

plt.xlabel("Velocity (m/s)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Coasting Curve Fit")
plt.legend()
plt.grid()

plt.show()
