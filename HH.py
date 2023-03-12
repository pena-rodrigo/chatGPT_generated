import numpy as np
import matplotlib.pyplot as plt

# Define parameters
C_m  =   1.0 # membrane capacitance (uF/cm^2)
g_Na = 120.0 # maximum conductances (mS/cm^2)
g_K  =  36.0
g_l  =   0.3
E_Na =  50.0 # Nernst reversal potentials (mV)
E_K  = -77.0
E_l  = -54.387
I    =   5.0 # input current (uA/cm^2)

# Define simulation time
t_start = 0
t_stop = 50
dt = 0.01
t = np.arange(t_start, t_stop, dt)

# Initialize variables
V = np.zeros(len(t))
m = np.zeros(len(t))
h = np.zeros(len(t))
n = np.zeros(len(t))
V[0] = -65.0
m[0] = 0.0529
h[0] = 0.5961
n[0] = 0.3177

# Define gating variables
def alpha_m(V): return 0.1*(V+40.0)/(1.0-np.exp(-(V+40.0)/10.0))
def beta_m(V):  return 4.0*np.exp(-(V+65.0)/18.0)
def alpha_h(V): return 0.07*np.exp(-(V+65.0)/20.0)
def beta_h(V):  return 1.0/(1.0+np.exp(-(V+35.0)/10.0))
def alpha_n(V): return 0.01*(V+55.0)/(1.0-np.exp(-(V+55.0)/10.0))
def beta_n(V):  return 0.125*np.exp(-(V+65)/80.0)

# Simulate the neuron
for i in range(1, len(t)):
    # Calculate gating variables
    m[i] = m[i-1] + dt*(alpha_m(V[i-1])*(1.0-m[i-1])-beta_m(V[i-1])*m[i-1])
    h[i] = h[i-1] + dt*(alpha_h(V[i-1])*(1.0-h[i-1])-beta_h(V[i-1])*h[i-1])
    n[i] = n[i-1] + dt*(alpha_n(V[i-1])*(1.0-n[i-1])-beta_n(V[i-1])*n[i-1])

    # Calculate membrane potential
    g_Na_m = g_Na * m[i-1]**3 * h[i-1]
    g_K_n = g_K * n[i-1]**4
    g_l_ = g_l
    I_Na = g_Na_m * (V[i-1] - E_Na)
    I_K = g_K_n * (V[i-1] - E_K)
    I_l = g_l_ * (V[i-1] - E_l)
    V[i] = V[i-1] + (I - I_Na - I_K - I_l) / C_m * dt

# Plot the results
plt.plot(t, V)
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential (mV)')
plt.show()
