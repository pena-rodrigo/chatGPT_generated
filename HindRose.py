import numpy as np
import matplotlib.pyplot as plt

# Define parameters
a = 1
b = 3
c = 1
d = 5
r = 0.001
s = 4
x_r = -1.6
I = 3.1
v_th = 0

# Define simulation time
t_start = 0
t_stop = 3000
dt = 0.01
t = np.arange(t_start, t_stop, dt)

# Initialize variables
x = np.zeros(len(t))
y = np.zeros(len(t))
z = np.zeros(len(t))
x[0] = -1.4
y[0] = 1

# Simulate the neuron
for i in range(1, len(t)):
    # Calculate membrane potential and recovery variable
    dx_dt = y[i-1] - a * x[i-1]**3 + b * x[i-1]**2 - z[i-1] + I
    dy_dt = c - d * x[i-1]**2 - y[i-1]
    dz_dt = r * (s * (x[i-1] - x_r) - z[i-1])
    x[i] = x[i-1] + dx_dt * dt
    y[i] = y[i-1] + dy_dt * dt
    z[i] = z[i-1] + dz_dt * dt

    # Check for spike
    if x[i] >= v_th:
        x[i] = -1.6
        y[i] = y[i] + 1.0

# Plot the results
plt.plot(t, x)
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential')
plt.show()
