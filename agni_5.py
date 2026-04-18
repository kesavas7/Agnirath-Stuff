import numpy as np
#environment purely made with the help of chatgpt
#in this environment vmax is not found by solving the equation we defined
#rather just assumed as 100

T = 1.5              # hours
L = 30               # km
E_current = 1.5      # kWh
E_min = 0.2          # kWh
P_solar = 0.25       # kW (250 W)

E_avail = E_current - E_min

eta = 0.9            # motor efficiency (already discussed)
P_losses = 0.15      # kW (150 W) average power loss of a car is around 150W
k = 0.00002          # kW/(km/h)^3  average drag force for a car is 0.00002


def energy_used(v, N):

    P_mech = k * v**3 + P_losses
    P_draw = P_mech / eta
    P_net = P_draw - P_solar
    
    t = (N * L) / v  # hours
    
    return P_net * t

def find_vmax(N, v_min, v_upper=100, num_points=1000):
    speeds = np.linspace(v_min, v_upper, num_points)
    feasible_speeds = []
    
    for v in speeds:
        if energy_used(v, N) <= E_avail:
            feasible_speeds.append(v)
    
    if not feasible_speeds:
        return None
    
    return max(feasible_speeds)

results = []

for N in range(1, 10):  # try up to 9 loops
    v_min = (N * L) / T
    
    vmax = find_vmax(N, v_min)
    
    if vmax is not None and vmax >= v_min:
        results.append((N, v_min, vmax))

if not results:
    print("No feasible solution")
else:
    best = max(results, key=lambda x: x[0])  # max N
    
    print("All feasible options:")
    for r in results:
        print(f"N = {r[0]} | v_min = {r[1]:.2f} km/h | v_max = {r[2]:.2f} km/h")
    
    print("\nBest choice:")
    print(f"Max loops N = {best[0]}")
    print(f"Speed range: {best[1]:.2f} to {best[2]:.2f} km/h")
    
    # Suggested strategy (lower end of range)
    v_target = best[1] + 5
    print(f"Recommended target speed ≈ {v_target:.2f} km/h")
