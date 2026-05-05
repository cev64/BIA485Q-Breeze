# Air Quality RL Agent

An autonomous reinforcement learning agent that learns when to issue air quality alerts based on real sensor readings, replacing rigid static thresholds with a dynamic learned policy.

---

## Overview

**Topic:** Reinforcement Learning  
**Algorithm:** Q-Learning (Value-Based RL)  
**Dataset:** [UCI Air Quality Dataset](https://archive.ics.uci.edu/dataset/360/air+quality)

Current air quality alert systems rely on fixed thresholds — if CO exceeds X ppm, trigger an alert. This approach is brittle: it ignores combinations of pollutants, sensor drift, and time-of-day patterns. This project trains a Q-learning agent to learn a **dynamic alerting policy** from real sensor data.

---

## Actions

At each hourly timestep, the agent observes sensor readings and decides:

| Action | Description |
|--------|-------------|
| 0 | No alert — air quality is safe |
| 1 | Issue a moderate warning |
| 2 | Issue a severe alert / trigger ventilation |

Alert levels are grounded in WHO guidelines for CO concentration (mg/m³):
- **Safe:** CO < 4
- **Moderate:** CO 4–10
- **Severe:** CO > 10

---

## Project Structure

```
├── air_quality_rl_project.ipynb   # Main notebook (full pipeline)
├── AirQualityUCI.csv              # Dataset
├── rl.pdf                         # RL background reference slides
└── README.md
```

---

## Pipeline

1. **Data Cleaning** — Interpolate -200 sentinel values across all sensor columns
2. **EDA** — Visualize pollutant distributions, hourly CO patterns, correlations
3. **Feature Engineering** — Add rolling averages, lag features, hour-of-day, rush hour flags
4. **RL Environment** — Custom environment with state discretization and reward shaping
5. **Q-Learning Training** — 1000 episodes with epsilon-greedy exploration and decay
6. **Hyperparameter Tuning** — Compare learning rate, discount factor, and decay configurations
7. **Evaluation** — Policy accuracy, cumulative reward, confusion matrix, missed severe events

---

## Results

| Metric | Value |
|--------|-------|
| Final Policy Accuracy | 97% |
| Total Cumulative Reward | 18,172 |
| Missed Severe Events | 0 |
| False Alarms (Safe → Severe) | 0 |
| Best Config | α=0.1, γ=0.5, slow decay |

---

## Dependencies

```
pip install pandas numpy matplotlib seaborn scikit-learn ucimlrepo
```

---

## Dataset

**UCI Air Quality Dataset** — 9,357 hourly readings from a gas multisensor device deployed in an Italian city.  
Source: https://archive.ics.uci.edu/dataset/360/air+qualityy

Key columns used: `CO(GT)`, `PT08.S1(CO)`, `C6H6(GT)`, `NOx(GT)`, `NO2(GT)`, `T`, `RH`, `AH`
