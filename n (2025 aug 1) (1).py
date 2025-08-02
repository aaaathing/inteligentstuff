import torch
from torch import tensor
from time import sleep


class Path:
	def __init__(self, sender, reciever):
		self.sender = sender
		self.reciever = reciever
		self.gExciteRaw = torch.zeros(reciever.size)
		# each neuron connects to many neurons
		self.weight = None # shape (reciever.size, sender.size)
	def gatherInputs(self):
		gExciteRaw = (self.sender.sentSpike * self.weight).to_dense().sum(1)
		self.gExciteRaw = gExciteRaw #self.gExciteRaw*(1-GeDt) + gExciteRaw
		self.reciever.gExciteRaw += self.gExciteRaw

paths = []
def connectAllToAll(sender, reciever):
	p = Path(sender,reciever)
	p.weight = (torch.rand((reciever.size,sender.size)) * 0.1 + 0.4).to_sparse_coo()
	reciever.paths.append(p)
	paths.append(p)
	return p


class Pools:
	def __init__(self):
		self.inhibitPotential = torch.zeros(1)
		self.inhibitSpike = torch.zeros(1)
	def inhibit(self, layer):
		self.inhibitPotential += layer.spike.sum(0)/layer.size
		self.inhibitSpike = self.inhibitPotential>0.9
		self.inhibitPotential[self.inhibitSpike] = 0


class Layer:
	def __init__(self, size: int):
		self.size = size
		self.paths = []
		self.gExciteRaw = torch.zeros(size)
		self.gInhibitRaw = torch.zeros(size)
		self.gExcite = torch.zeros(size) # these reset every time
		self.gInhibit = torch.zeros(size)
		self.potential = torch.zeros(size)
		self.spike = torch.zeros(size)
		self.sentSpike = torch.zeros(size)
		self.pools = Pools()

	def gatherInputs(self):
		self.gExciteRaw.zero_()
		self.gInhibitRaw.zero_()

		for path in self.paths:
			assert path.reciever == self
			path.gatherInputs()
		
		self.gExcite = self.gExciteRaw
		# todo: add noise to gExcite
		self.gInhibit = self.gInhibitRaw
	def layerInhibit(self):
		self.pools.inhibit(self)
		self.gInhibit += self.pools.inhibitSpike
	def update(self):
		self.potential += self.gExcite
		self.potential -= self.gInhibit
		self.potential.clamp_min_(0)
		self.spike = self.potential > 0.9
		self.potential[self.spike] = 0
	def sendOutput(self):
		self.sentSpike = self.spike

layers = []
def addLayer(l):
	layers.append(l)
	return l

import matplotlib.pyplot as plt
import matplotlib
plts = [plt.subplot(2,3,1),plt.subplot(2,3,2),plt.subplot(2,3,3),plt.subplot(2,3,4),plt.subplot(2,3,5),plt.subplot(2,3,6)]
prog1=[];prog2=[];prog3=[]


l1=addLayer(Layer(5))
l2=addLayer(Layer(6))
#l3=addLayer(Layer(6))
#l4=addLayer(Layer(5))
connectAllToAll(l1,l2)
#connectAllToAll(l2,l3)
#l4.paths.append(Path(l3,l4))
for i in range(1,30):
	l1.sentSpike = tensor([1,0,0,0,0])
	for l in layers: l.gatherInputs()
	for l in layers: l.layerInhibit()
	for l in layers: l.update()
	for l in layers: l.sendOutput()
	#print(l2.potential,l2.gInhibit[0])
	plts[0].imshow([l2.gExcite], interpolation='nearest', vmin=0, vmax=1)
	plts[1].imshow([l2.gInhibit], interpolation='nearest', vmin=0, vmax=1)
	plts[2].imshow([l2.potential], interpolation='nearest', vmin=0, vmax=1)
	prog1.append(tensor(l2.gExcite)); plts[3].imshow(prog1, interpolation='nearest', vmin=0, vmax=1)
	prog2.append(tensor(l2.gInhibit)); plts[4].imshow(prog2, interpolation='nearest', vmin=0, vmax=1)
	prog3.append(tensor(l2.potential)); plts[5].imshow(prog3, interpolation='nearest', vmin=0, vmax=1)
	plt.pause(1)
#print(l2.paths[0].weight.to_dense())

plt.show()
