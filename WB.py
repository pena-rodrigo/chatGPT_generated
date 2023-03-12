import numpy as np
import matplotlib.pyplot as plt

# Define parameters
C = 1.0
gL = 0.1
EL = -60.0
VT = -50.0
DeltaT = 2.0
a = 0.01
b = 5.0
Vr = -60.0
Vp = 35.0
I = 1.5

# Define simulation time
t_start = 0
t_stop = 200
dt = 0.1
t = np.arange(t_start, t_stop, dt)

# Initialize variables
V = np.zeros(len(t))
U = np.zeros(len(t))
V[0] = EL

# Simulate the neuron
for i in range(1, len(t)):
    # Calculate membrane potential and recovery variable
    dv_dt = (gL * (EL - V[i-1]) + gL * DeltaT * np.exp((V[i-1] - VT) / DeltaT) - U[i-1] + I) / C
    du_dt = a * (b * (V[i-1] - EL) - U[i-1])
    V[i] = V[i-1] + dv_dt * dt
    U[i] = U[i-1] + du_dt * dt

    # Check for spike
    if V[i] >= Vp:
        V[i] = Vr
        U[i] = U[i] + b

# Plot the results
plt.plot(t, V)
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential (mV)')
plt.show()
