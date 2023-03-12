import numpy as np
import matplotlib.pyplot as plt

# Define parameters
C = 281.0 # pF
gL = 30.0 # nS
EL = -70.6 # mV
VT = -50.4 # mV
DeltaT = 2.0 # mV
tauw = 40.0 # ms
a = 4.0 # nS
b = 0.0805 # nA
Vr = -70.6 # mV
Vth = -50.4 # mV
I = 0.0 # nA

# Define simulation time
t_start = 0
t_stop = 500
dt = 0.01
t = np.arange(t_start, t_stop, dt)

# Initialize variables
V = np.zeros(len(t))
w = np.zeros(len(t))
V[0] = -70.6
w[0] = 0.0

# Simulate the neuron
for i in range(1, len(t)):
    # Calculate membrane potential
    dV_dt = (gL * (EL - V[i-1]) + gL * DeltaT * np.exp((V[i-1] - VT) / DeltaT) - w[i-1] + I) / C
    V[i] = V[i-1] + dV_dt * dt

    # Calculate adaptation variable
    dw_dt = (a * (V[i-1] - EL) - w[i-1]) / tauw
    w[i] = w[i-1] + dw_dt * dt

    # Check for spike
    if V[i] >= Vth:
        V[i-1] = Vth
        V[i] = Vr
        w[i] += b

# Plot the results
plt.plot(t, V)
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential (mV)')
plt.show()
