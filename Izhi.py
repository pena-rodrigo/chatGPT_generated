import numpy as np
import matplotlib.pyplot as plt

# Define parameters
a = 0.02    # Time scale of recovery variable
b = 0.2     # Sensitivity of recovery variable
c = -65     # Reset value of membrane potential
d = 8       # Reset value of recovery variable
v_th = 30   # Spike threshold
v_peak = 35 # Peak voltage reached during spike
I = 10      # Injected current

# Define simulation time
t_start = 0
t_stop = 500
dt = 0.1
t = np.arange(t_start, t_stop, dt)

# Initialize variables
v = c * np.ones(len(t))
u = 0

# Simulate the neuron
for i in range(1, len(t)):
    # Calculate membrane potential and recovery variable
    dv_dt = 0.04 * v[i-1]**2 + 5 * v[i-1] + 140 - u[i-1] + I
    du_dt = a * (b * v[i-1] - u[i-1])
    v[i] = v[i-1] + dv_dt * dt
    u[i] = u[i-1] + du_dt * dt

    # Check for spike
    if v[i] >= v_th:
        v[i] = c
        u[i] = u[i] + d

# Plot the results
plt.plot(t, v)
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential (mV)')
plt.show()
