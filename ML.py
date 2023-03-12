import numpy as np
import matplotlib.pyplot as plt

# Define parameters
C = 20.0
gCa = 4.4
gK = 8.0
gL = 2.0
VCa = 120.0
VK = -84.0
VL = -60.0
phi = 0.04
I = 80.0

# Define simulation time
t_start = 0
t_stop = 100
dt = 0.01
t = np.arange(t_start, t_stop, dt)

# Initialize variables
V = np.zeros(len(t))
n = np.zeros(len(t))
V[0] = -50.0
n[0] = 0.1

# Simulate the neuron
for i in range(1, len(t)):
    # Calculate gating variables
    alpha_n = 0.5 * (1 + np.tanh((V[i-1] - 10.0) / 10.0))
    beta_n = phi * np.cosh((V[i-1] - 10.0) / 40.0)

    # Calculate membrane potential
    m_inf = 0.5 * (1 + np.tanh((V[i-1] - VCa) / 15.0))
    w_inf = 0.5 * (1 + np.tanh((V[i-1] - VK + 30.0) / 15.0))
    tau_w = 5 / np.cosh((V[i-1] - VK + 30.0) / 60.0)
    dV_dt = (I - gCa * m_inf * (V[i-1] - VCa) - gK * n[i-1] * (V[i-1] - VK) - gL * (V[i-1] - VL)) / C
    dn_dt = (alpha_n * (1 - n[i-1]) - beta_n * n[i-1]) / tau_w
    V[i] = V[i-1] + dV_dt * dt
    n[i] = n[i-1] + dn_dt * dt

# Plot the results
plt.plot(t, V)
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential (mV)')
plt.show()
