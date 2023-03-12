import numpy as np
import matplotlib.pyplot as plt

# Define parameters
a = 0.7   # Time constant of recovery variable
b = 0.8   # Slope of nullcline
I = 0.5   # Injected current
v_th = 1  # Spike threshold

# Define simulation time
t_start = 0
t_stop = 200
dt = 0.01
t = np.arange(t_start, t_stop, dt)

# Initialize variables
v = np.zeros(len(t))
w = np.zeros(len(t))
v[0] = -1.2

# Simulate the neuron
for i in range(1, len(t)):
    # Calculate membrane potential and recovery variable
    dv_dt = v[i-1] - v[i-1]**3/3 - w[i-1] + I
    dw_dt = (v[i-1] + a - b * w[i-1]) / a
    v[i] = v[i-1] + dv_dt * dt
    w[i] = w[i-1] + dw_dt * dt

    # Check for spike
    if v[i] >= v_th:
        v[i] = 1
        w[i] = w[i-1] + b

# Plot the results
plt.plot(t, v)
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential')
plt.show()
