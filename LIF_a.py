import numpy as np
import matplotlib.pyplot as plt

# Define parameters
tau_m = 10      # Membrane time constant in ms
R_m = 10        # Membrane resistance in Mohms
V_th = -55      # Threshold voltage in mV
V_reset = -70   # Reset voltage in mV
I_inj = 2       # Injected current in nA
tau_a = 100     # Adaptation time constant in ms
b = 0.1         # Spike-triggered adaptation factor in nA

# Define simulation time
t_start = 0     # Start time in ms
t_stop = 500    # Stop time in ms
dt = 0.1        # Time step in ms
t = np.arange(t_start, t_stop, dt)

# Initialize variables
V = np.zeros(len(t))
V[0] = V_reset
a = np.zeros(len(t))

# Simulate the neuron
for i in range(1, len(t)):
    # Calculate membrane potential
    dV_dt = (-V[i-1] + R_m * (I_inj + a[i-1])) / tau_m
    V[i] = V[i-1] + dV_dt * dt
    
    # Calculate adaptation current
    da_dt = (-a[i-1] + b * (V[i-1] - V_reset)) / tau_a
    a[i] = a[i-1] + da_dt * dt
    
    # Check for spike
    if V[i] >= V_th:
        V[i] = V_reset
        a[i] += b * (V[i-1] - V_reset)

# Plot the results
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Membrane potential (mV)', color=color)
ax1.plot(t, V, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('Adaptation current (nA)', color=color)
ax2.plot(t, a, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()
