import torch
import matplotlib.pyplot as plt
import time

from bindsnet.network import Network
from bindsnet.network.nodes import Nodes, Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_voltages


network = Network()

# Create two layers.
layerA = LIFNodes(n=100)
layerB = LIFNodes(n=100)

# Add layers to the network.
network.add_layer(layerA, name="LayerA")
network.add_layer(layerB, name="LayerB")

# Create a connection from LayerA to LayerB.
connection_A_to_B = Connection(source=layerA, target=layerB)
network.add_connection(connection_A_to_B, source="LayerA", target="LayerB")

# Create a connection from LayerB to LayerA.
connection_B_to_A = Connection(source=layerB, target=layerA)
network.add_connection(connection_B_to_A, source="LayerB", target="LayerA")

network.add_monitor(Monitor(network.layers['LayerA'], state_vars=['s','v']), 'LayerB')

# Generate spikes by running Bernoulli trials on Uniform(0, 0.5) samples.
spikes = torch.bernoulli(torch.rand(1,100).repeat(200,1))

start=time.perf_counter(); print("start")

# Run network simulation.
network.run(inputs={'LayerA' : spikes}, time=200)

print("run time",time.perf_counter()-start)

# Look at input spiking activity.
plot_spikes({'LayerB':network.monitors['LayerB'].get('s')})
plot_voltages({'LayerB':network.monitors['LayerB'].get('v')}, plot_type="line")

plt.show(block=True)
