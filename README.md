# Agnirath-Stuff
This repo contains all the work i have done for strategy vertical application in agnirath comp team iitm
for the last section i have given code as 5 sperate parts main.py, base_optimizer.py, loop_optimizer.py, physice.py, plot_results.py
# 🚗 Agnirath — Day 2 Solar Strategy Optimization

### Sasol Solar Challenge | Final Challenge Submission

---

## 📌 Overview

This project implements a **full-day race strategy optimizer** for a solar-powered vehicle competing in the Sasol Solar Challenge.

The objective is to:

> **Maximize total distance traveled in a day**
> (Base route + post-arrival loops)
> while maintaining **battery State of Charge (SOC) ≥ 20%** at all times.

---

## 🧠 Core Idea

Instead of minimizing travel time, this solution:

> **Minimizes total battery energy consumption on the base route**
> → preserving energy for additional loops
> → maximizing total race distance

---

## ⚙️ System Architecture

The pipeline follows a modular design:

```
Route Data → Physics Model → Base Optimization → Loop Optimization → Visualization
```

---

## 📂 Project Structure

```
.
├── main.py              # Entry point: runs full pipeline
├── physics.py          # Physics + solar model
├── base_optimizer.py   # Base route optimization (SLSQP)
├── loop_optimizer.py   # Loop phase optimization
├── plot_results.py     # Visualization and analysis
├── route_data.csv      # Preprocessed route data
└── README.md
```

---

## 🔬 Physics Model

The system models realistic vehicle dynamics:

### Forces considered:

* Aerodynamic drag ∝ v³
* Rolling resistance
* Gravitational force (based on slope)

### Energy system:

* Battery capacity: **3.1 kWh**
* Motor efficiency: **90%**
* Regenerative braking: **70%**

---

## ☀️ Solar Model

Solar irradiance is modeled using a **Gaussian distribution**:

* Peak: **1073 W/m² at 12:00 PM**
* Standard deviation: **11,600 seconds**

Solar power:

```
P_solar = irradiance × panel_area × efficiency
```

* Panel area: **6 m²**
* Efficiency: **24%**

---

## 🛣️ Phase 1: Base Route Optimization

### Objective:

Minimize total battery energy consumed:

```
minimize ∑ energy_segment(v_i, slope_i, distance_i, time_i)
```

### Variables:

* Velocity at each route segment

### Constraints:

* SOC ≥ 20% throughout
* SOC at arrival ≥ target
* Arrival before 5:00 PM
* Velocity bounds: 60–100 km/h

### Method:

* Solver: **SciPy SLSQP**
* Iterates over multiple SOC targets (20%–60%)
* Selects best strategy based on:

  * Maximum loops achievable
  * Final SOC closest to 20%

---

## 🔁 Phase 2: Loop Optimization

After reaching Zeerust:

* 30-minute mandatory control stop (solar charging)
* Perform multiple **35 km loops**

### Constraints:

* 5-minute stop between loops
* Finish before 5:00 PM
* Final SOC ≥ 20%

### Key Insight:

> Optimal loop velocity = **minimum feasible speed**

Reason:

* Drag ∝ v³ → slower speeds reduce energy loss
* More time under sunlight → higher solar gain

---

## 📊 Phase 3: Visualization

Generates:

* Velocity vs Time
* SOC vs Time
* Acceleration vs Time
* Solar Power vs Time

### Special Feature:

* **Solar charging during stops is explicitly modeled**
* SOC increases during:

  * Control stop
  * Inter-loop stops

---

## 🚀 Key Innovations

### ✅ Energy-first optimization

* Prioritizes energy efficiency over speed

### ✅ Two-stage optimization

* Complex terrain handled separately from loops

### ✅ Realistic solar integration

* Charging during stationary periods included

### ✅ Constraint-aware system

* Ensures feasibility at all times

---

## ⚡ Performance Optimizations

To reduce runtime (~1 hour → ~10–15 minutes):

* Route downsampling (reduces variables)
* Reduced SOC target search space
* Lower optimizer iterations
* Faster loop velocity search
* Simplified constraint evaluation

---

## 📈 Results

The system outputs:

* Optimal velocity profile
* Arrival time and SOC
* Number of loops completed
* Total distance traveled
* Final SOC

---

## 🧪 Assumptions

* Loop terrain is flat
* Solar irradiance follows ideal Gaussian curve
* No weather variability (clouds, shading)
* Constant vehicle parameters

---

## ⚠️ Limitations

* No acceleration constraints in optimization
* No stochastic solar variation
* Loop phase ignores elevation
* Solver may converge to local minima

---

## 🔮 Future Improvements

* Add acceleration constraints
* Incorporate weather variability
* Use dynamic programming for faster optimization
* Introduce machine learning for speed prediction
* Multi-objective optimization (time vs energy)

---

## ▶️ How to Run

```bash
python main.py
```

Output:

* Console summary
* `full_day_strategy.png`

---

## 🏁 Conclusion

This solution demonstrates a **physics-driven, optimization-based race strategy system** that balances:

* Energy consumption
* Solar gain
* Time constraints

to maximize total race distance under realistic conditions.

---
