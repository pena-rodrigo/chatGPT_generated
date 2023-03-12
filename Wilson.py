import numpy as np
import matplotlib.pyplot as plt

# Define parameters
C = 1.0
gL = 0.3
VL = -80.0
VT = -57.0
DeltaT = 2.0
a = 0.0
b = 100.0
tauw = 30.0
I = 100.0

# Define simulation time
t_start = 0
t_stop = 100
dt = 0.01
t = np.arange(t_start, t_stop, dt)

# Initialize variables
V = np.zeros(len(t))
w = np.zeros(len(t))
V[0] = -70.0
w[0] = 0.0

# Simulate the neuron
for i in range(1, len(t)):
    # Calculate gating variable
    alpha_w = (0.04 * (V[i-1] - VT) + 5) / (np.exp((V[i-1] - VT) / 5) - 1)
    beta_w = a * (V[i-1] - VT)

    # Calculate membrane potential
    dV_dt = (-gL * (V[i-1] - VL) + gL * DeltaT * np.exp((V[i-1] - VT) / DeltaT) - w[i-1] + I) / C
    dw_dt = (alpha_w * (1 - w[i-1]) - beta_w * w[i-1]) / tauw
    V[i] = V[i-1] + dV_dt * dt
    w[i] = w[i-1] + dw_dt * dt

    # Check for spike
    if V[i] > 0:
        V[i-1] = 30.0
        V[i] = VL
        w[i] = w[i] + b

# Plot the results
plt.plot(t, V)
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential (mV)')
plt.show()
