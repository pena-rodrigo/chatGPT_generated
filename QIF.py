import numpy as np
import matplotlib.pyplot as plt

# Define parameters
tau_m = 10      # Membrane time constant in ms
R_m = 10        # Membrane resistance in Mohms
V_th = -55      # Threshold voltage in mV
V_reset = -70   # Reset voltage in mV
I_inj = 2       # Injected current in nA
a = 0.001       # Quadratic term coefficient in mV^-1

# Define simulation time
t_start = 0     # Start time in ms
t_stop = 500    # Stop time in ms
dt = 0.1        # Time step in ms
t = np.arange(t_start, t_stop, dt)

# Initialize variables
V = np.zeros(len(t))
V[0] = V_reset

# Simulate the neuron
for i in range(1, len(t)):
    # Calculate membrane potential
    dV_dt = (-V[i-1] + R_m * I_inj + a * (V[i-1] - V_reset)**2) / tau_m
    V[i] = V[i-1] + dV_dt * dt
    
    # Check for spike
    if V[i] >= V_th:
        V[i] = V_reset

# Plot the results
plt.plot(t, V)
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential (mV)')
plt.show()
