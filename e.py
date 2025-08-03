import numpy as np
import matplotlib.pyplot as plt

# Constants
Cm = 1.0  # Membrane capacitance in uF/cm^2
gNa = 120.0  # Sodium channel conductance in mS/cm^2
gK = 36.0  # Potassium channel conductance in mS/cm^2
gL = 0.3  # Leak conductance in mS/cm^2
ENa = 50.0  # Sodium reversal potential in mV
EK = -77.0  # Potassium reversal potential in mV
EL = -54.387  # Leak reversal potential in mV
I_ext = 10.0  # External current in uA/cm^2

# Time vector
t = np.linspace(0, 50, 1000)  # Time from 0 to 50 ms, 10000 points

# Initial conditions
potential = -65.0  # Membrane potential in mV
sodiumActive = 0.05  # Activation of sodium channels
sodiumInactive = 0.6  # Inactivation of sodium channels
potassiumActive = 0.32  # Activation of potassium channels

# Functions for gating variables
def alpha_m(potential): return 0.1*(potential+40)/(1-np.exp(-(potential+40)/10))
def beta_m(potential): return 4.0*np.exp(-(potential+65)/18)
def alpha_h(potential): return 0.07*np.exp(-(potential+65)/20)
def beta_h(potential): return 1.0/(1+np.exp(-(potential+35)/10))
def alpha_n(potential): return 0.01*(potential+55)/(1-np.exp(-(potential+55)/10))
def beta_n(potential): return 0.125*np.exp(-(potential+65)/80)

# Differential equations for Hodgkin-Huxley model
def dVdt(potential, sodiumActive, sodiumInactive, potassiumActive):
    I_Na = gNa*(sodiumActive**3)*sodiumInactive*(potential-ENa)
    I_K = gK*(potassiumActive**4)*(potential-EK)
    I_L = gL*(potential-EL)
    I_ion = I_Na + I_K + I_L
    return (I_ext - I_ion) / Cm

def dmdt(potential, sodiumActive): return alpha_m(potential)*(1-sodiumActive) - beta_m(potential)*sodiumActive
def dhdt(potential, sodiumInactive): return alpha_h(potential)*(1-sodiumInactive) - beta_h(potential)*sodiumInactive
def dndt(potential, potassiumActive): return alpha_n(potential)*(1-potassiumActive) - beta_n(potential)*potassiumActive

# Time-stepping loop
V_trace, m_trace, h_trace, n_trace = [], [], [], []
for time in t:
    V_trace.append(potential)
    m_trace.append(sodiumActive)
    h_trace.append(sodiumInactive)
    n_trace.append(potassiumActive)
    
    potential += dVdt(potential, sodiumActive, sodiumInactive, potassiumActive) * (t[1] - t[0])
    sodiumActive += dmdt(potential, sodiumActive) * (t[1] - t[0])
    sodiumInactive += dhdt(potential, sodiumInactive) * (t[1] - t[0])
    potassiumActive += dndt(potential, potassiumActive) * (t[1] - t[0])

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(t, V_trace, label='Membrane Potential (mV)')
plt.title('Hodgkin-Huxley Action Potential')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.grid(True)
plt.legend()
plt.show()